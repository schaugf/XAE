from __future__ import print_function
import os
import argparse
from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
#from torchsummary import summary


parser = argparse.ArgumentParser(description='XAE Model')
parser.add_argument('--batch_size', type=int, default=12, metavar='N',
                    help='input batch size for training (default: 12)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='optimizer learning rate (default: 1e-3)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--latent_dim', type=int, default=20, metavar='N',
                    help='how big is z')
parser.add_argument('--inter_dim', type=int, default=128, metavar='N',
                    help='how big is linear around z')
parser.add_argument('--dataset', type=str, default='test', metavar='N',
                    help='dataset to load (mnist or test)')
parser.add_argument('--save_dir', type=str, default='../results/test', 
                    metavar='N', help='save directory')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging status')
args = parser.parse_args()


args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# load train and test
train_mnist = datasets.MNIST('../data', 
                             train = True, 
                             download = True,
                             transform = transforms.ToTensor())

test_mnist = datasets.MNIST('../data', 
                            train = False, 
                            download = True, 
                            transform = transforms.ToTensor())

train_img = train_mnist.train_data.unsqueeze(1).float()/255
train_ome = train_img.reshape((len(train_img), -1))

test_img = test_mnist.test_data.unsqueeze(1).float()/255
test_ome = test_img.reshape((len(test_img), -1))

if args.dataset == 'test':
    train_img = train_img[:200]
    train_ome = train_ome[:200]
    test_img = test_img[:200]
    test_ome = test_ome[:200]

train = torch.utils.data.TensorDataset(train_img.float(), train_ome.float())

train_loader =  torch.utils.data.DataLoader(train, 
                                            batch_size = args.batch_size, 
                                            shuffle = True)

eval_loader = torch.utils.data.DataLoader(train, 
                                          batch_size = args.batch_size, 
                                          shuffle = False)

test = torch.utils.data.TensorDataset(test_img, test_ome)
test_loader =  torch.utils.data.DataLoader(test, 
                                           batch_size = args.batch_size, 
                                           shuffle = False)

# TODO: add domain corruption
# TODO: save original omic domain shape (length)

# set data parameters
A_type = 'img'
B_type = 'ome'
A_shape = train_img.shape[1:]
B_shape = train_ome.shape[1:]



#def AddDomainCorruption(self):
#        ''' append domain-specific corruption '''
#    
#        print('corrupting omics domain')
#        n_samples_to_add = int(self.ome_train.shape[1] * 
#                               self.test_rand_add /
#                               (1 - self.test_rand_add))
#        
#        train_shape_to_add = (self.ome_train.shape[0], n_samples_to_add)
#        test_shape_to_add = (self.ome_test.shape[0], n_samples_to_add)
#        
#        sample_space = np.reshape(self.ome_train, -1)
#
#        train_samples = np.random.choice(sample_space, 
#                                         np.prod(train_shape_to_add))
#        
#        test_samples = np.random.choice(sample_space, 
#                                        np.prod(test_shape_to_add))
#        
#        train_samples = np.reshape(train_samples, train_shape_to_add)
#        test_samples = np.reshape(test_samples, test_shape_to_add)
#    
#        self.ome_train = np.concatenate((self.ome_train, train_samples), 
#                                        axis = 1)
#        
#        self.ome_test = np.concatenate((self.ome_test, test_samples), 
#                                        axis = 1)
#        
#        print('corrupted omic train shape', self.ome_train.shape)
#        print('corrupted omic test shape', self.ome_test.shape)
        


# define shared XAE layer
class XAE(nn.Module):
    def __init__(self, 
                 A_type = 'img', 
                 B_type = 'ome',
                 A_shape = (),
                 B_shape = (),
                 latent_dim = 8,
                 inter_dim = 32):
        
        super(XAE, self).__init__()
        
        self.A_type = A_type
        self.B_type = B_type
        self.A_shape = A_shape
        self.B_shape = B_shape
        self.latent_dim = latent_dim
        self.inter_dim = inter_dim
        
        # TODO: implement PWGL
        
        # shared latent feature layers
        self.shared_fc1 = nn.Linear(self.inter_dim, self.latent_dim)
        self.shared_fc2 = nn.Linear(self.inter_dim, self.latent_dim)
        
        # configure A-domain
        if self.A_type == 'img':
            # load modules from definitions
            self.A_encoder = self.make_image_encoder(in_shape = A_shape)
            self.A_decoder = self.make_image_decoder(out_shape = A_shape)
        elif self.A_type == 'ome':
            self.A_encoder = self.make_omic_encoder(in_shape = A_shape)
            self.A_decoder = self.make_omic_decoder(out_shape = A_shape)
        
        # configure B-domain
        if self.B_type == 'img':
            self.B_encoder = self.make_image_encoder(in_shape = B_shape)
            self.B_decoder = self.make_image_decoder(out_shape = B_shape)
        elif self.B_type == 'ome':
            self.B_encoder = self.make_omic_encoder(in_shape = B_shape)
            self.B_decoder = self.make_omic_decoder(out_shape = B_shape)
     
        
    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size()[0], -1)
        
        
    class Rachet(nn.Module):
        def __init__(self, out_shape):
            super().__init__()
            self.out_shape = out_shape
            
        def forward(self, x):
            return x.view(x.size(0), 8, self.out_shape[1], self.out_shape[2])


    def make_image_encoder(self, in_shape):
        # need: number of input channels, image size
        img_encoder = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                self.Flatten(),
                nn.Linear(32, self.inter_dim),
                nn.ReLU()
                )
        
        return img_encoder
    
    
    def make_image_decoder(self, out_shape):
        
        img_decoder = nn.Sequential(
                nn.Linear(self.latent_dim, self.inter_dim),
                nn.ReLU(),
                nn.Linear(self.inter_dim, 8*np.prod(out_shape[1:])),
                nn.ReLU(),
                self.Rachet(out_shape),
                nn.ConvTranspose2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, out_shape[0], kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
                )
        
        return img_decoder
    
    
    def make_omic_encoder(self, in_shape):
        
        # TODO: implement gate layer
        ome_encoder = nn.Sequential(
                nn.Linear(in_shape[0], self.inter_dim*8),
                nn.ReLU(),
                nn.Linear(self.inter_dim*8, self.inter_dim),
                nn.ReLU()
                )
        
        return ome_encoder
        
    def make_omic_decoder(self, out_shape):
        
        ome_decoder = nn.Sequential(
                nn.Linear(self.latent_dim, self.inter_dim),
                nn.ReLU(),
                nn.Linear(self.inter_dim, self.inter_dim*8),
                nn.ReLU(),
                nn.Linear(self.inter_dim*8, out_shape[0]),
                nn.Sigmoid()
                )
        
        return ome_decoder
        

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std


    def forward(self, A_in, B_in):
        
        # autoencoders
        A_inter = self.A_encoder(A_in)
        A_mu, A_logvar = self.shared_fc1(A_inter), self.shared_fc2(A_inter)
        A_z = self.reparameterize(A_mu, A_logvar)
        A_rec = self.A_decoder(A_z)
        
        B_inter = self.B_encoder(B_in)
        B_mu, B_logvar = self.shared_fc1(B_inter), self.shared_fc2(B_inter)
        B_z = self.reparameterize(B_mu, B_logvar)
        B_rec = self.B_decoder(B_z)
        
        # cross-domain generators
        A2B_pred = self.B_decoder(A_z)
        B2A_pred = self.A_decoder(B_z)
        
        # cycle autoencoders
        A2B_inter = self.B_encoder(A2B_pred)
        A2B_mu, A2B_logvar = self.shared_fc1(A2B_inter), self.shared_fc2(A2B_inter)
        A2B_z = self.reparameterize(A2B_mu, A2B_logvar)
        
        B2A_inter = self.A_encoder(B2A_pred)
        B2A_mu, B2A_logvar = self.shared_fc1(B2A_inter), self.shared_fc2(B2A_inter)
        B2A_z = self.reparameterize(B2A_mu, B2A_logvar)
        
        # cycle reconstructions
        A2B2A_rec = self.A_decoder(A2B_z)
        B2A2B_rec = self.B_decoder(B2A_z)
        
        return {'A_rec' : A_rec, 
                'B_rec' : B_rec, 
                'A_mu' : A_mu, 
                'A_logvar' : A_logvar, 
                'A_z' : A_z, 
                'B_mu' : B_mu, 
                'B_logvar' : B_logvar, 
                'B_z' : B_z, 
                'A2B_pred' : A2B_pred, 
                'B2A_pred' : B2A_pred, 
                'A2B_mu' : A2B_mu, 
                'A2B_logvar' : A2B_logvar, 
                'A2B_z' : A2B_z, 
                'B2A_mu' : B2A_mu, 
                'B2A_logvar' : B2A_logvar, 
                'B2A_z' : B2A_z,
                'A2B2A_rec' : A2B2A_rec,
                'B2A2B_rec' : B2A2B_rec}
        
    
model = XAE(A_type = 'img', 
            B_type = 'ome',
            A_shape = A_shape,
            B_shape = B_shape)

#summary(model, input_size=[A_shape, B_shape])


if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr = args.lr)


def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(-1, np.prod(recon_x.shape[1:])),
                                 x.view(-1, np.prod(x.shape[1:])), 
                                 reduction = 'sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def cycle_loss(recon_x, x):
    CC = F.binary_cross_entropy(recon_x.view(-1, np.prod(recon_x.shape[1:])), 
                                x.view(-1, np.prod(x.shape[1:])),
                                reduction = 'sum')
    return CC


def mutual_encoding_loss(z1, z2):
    MED = F.mse_loss(z1, z2)
    return MED
    

def train(epoch, is_final):
    model.train()
    loss_keys = ['A_vae_loss',
                 'B_vae_loss',
                 'A_cycle_loss',
                 'B_cycle_loss',
                 'A_med_loss',
                 'B_med_loss',
                 'total_loss']
    all_losses = []

    for batch_idx, (A_data, B_data) in enumerate(train_loader):
        A_data = Variable(A_data)
        B_data = Variable(B_data)
        
        if args.cuda:
            A_data = A_data.cuda()
            B_data = B_data.cuda()
            
        optimizer.zero_grad()
        
        # pass through the model
        return_dict = model(A_data, B_data)
        
        A_vae_loss = vae_loss(recon_x = return_dict['A_rec'], 
                              x = A_data, 
                              mu = return_dict['A_mu'], 
                              logvar = return_dict['A_logvar'])
        
        B_vae_loss = vae_loss(recon_x = return_dict['B_rec'], 
                              x = B_data, 
                              mu = return_dict['B_mu'], 
                              logvar = return_dict['B_logvar'])
        
        A_cycle_loss = cycle_loss(recon_x = return_dict['A2B2A_rec'],
                                  x = A_data)
        
        B_cycle_loss = cycle_loss(recon_x = return_dict['B2A2B_rec'],
                                  x = B_data)
        
        A_med_loss = mutual_encoding_loss(z1 = return_dict['A_z'], 
                                          z2 = return_dict['A2B_z'])
        
        B_med_loss = mutual_encoding_loss(z1 = return_dict['B_z'], 
                                          z2 = return_dict['B2A_z'])
                
        
        xae_loss = A_vae_loss + B_vae_loss + \
                   A_cycle_loss + B_cycle_loss + \
                   A_med_loss + B_med_loss
                   
        xae_loss.backward()
        optimizer.step()
        
        loss_vals = [A_vae_loss.item(),
                     B_vae_loss.item(),
                     A_cycle_loss.item(),
                     B_cycle_loss.item(),
                     A_med_loss.item(),
                     B_med_loss.item()]
        
        all_losses.append(dict(zip(loss_keys, loss_vals)))
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 
                batch_idx * len(A_data), 
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                xae_loss.data[0] / len(A_data)))
            
    
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, xae_loss / len(train_loader.dataset)))

    # append losses for each epoch to dataframe
    mean_loss = pd.DataFrame(pd.DataFrame(all_losses).mean()).T
    mean_loss['epoch'] = epoch
    
    with open(os.path.join(args.save_dir, 'xae_loss.csv'), 'a') as f:
        mean_loss.to_csv(f, 
                         header = [True if epoch == 1 else False][0], 
                         index = False)
    
    # save batch of reconstructions
    gt_imgs = A_data[:20,0,:].detach().numpy()
    gt_stack = np.concatenate(list(gt_imgs))
    
    img_rec = return_dict['A_rec'][:20,0,:].detach().numpy()
    rec_stack = np.concatenate(list(img_rec))
    
    img_cyc_rec = return_dict['A2B2A_rec'][:20,0,:].detach().numpy()
    cyc_rec_stack = np.concatenate(list(img_cyc_rec))
    
    A2B = return_dict['A2B_pred'][:20]
    A2B_stack = np.concatenate(list(A2B.reshape(8, 28, 28).detach().numpy()))
    
    if epoch == 1:
        imgs_to_save = np.concatenate((gt_stack,
                                       rec_stack,
                                       A2B_stack,
                                       cyc_rec_stack,
                                       np.ones((gt_stack.shape[0], 1))), axis=1)
    else:
        # load image from file
        tmp_img = Image.open(os.path.join(args.save_dir, 
                                          'image_reconstructions.png'))
        tmp_img = np.array(tmp_img) / 255
        imgs_to_save = np.concatenate((tmp_img,
                                       gt_stack,
                                       rec_stack,
                                       A2B_stack,
                                       cyc_rec_stack,
                                       np.ones((gt_stack.shape[0], 1))), axis=1)
    
    to_save = Image.fromarray((255*imgs_to_save).astype(np.uint8))
    to_save.save(os.path.join(args.save_dir, 'image_reconstructions.png'))
    

    # TODO: save VAE reconstructions
    # TODO: save cycle reconstructions
    # TODO: save transformations
    

def encode_all():
    
    for batch_idx, (A_data, B_data) in enumerate(eval_loader):
        A_data = Variable(A_data)
        B_data = Variable(B_data)
        
        if args.cuda:
            A_data = A_data.cuda()
            B_data = B_data.cuda()
            
        return_dict = model(A_data, B_data)
    
        # save encodings
        A_encodings = pd.DataFrame(return_dict['A_z'].detach().numpy())
        with open(os.path.join(args.save_dir, 'A_encodings.csv'), 'a') as f:
            A_encodings.to_csv(f, index=False, header=False)
        
        B_encodings = pd.DataFrame(return_dict['B_z'].detach().numpy())
        with open(os.path.join(args.save_dir, 'B_encodings.csv'), 'a') as f:
            B_encodings.to_csv(f, index=False, header=False)
        
        # save cycle-encodings
        A2B_encodings = pd.DataFrame(return_dict['A2B_z'].detach().numpy())
        with open(os.path.join(args.save_dir, 'A2B_encodings.csv'), 'a') as f:
            A2B_encodings.to_csv(f, index=False, header=False)
        
        B2A_encodings = pd.DataFrame(return_dict['B2A_z'].detach().numpy())
        with open(os.path.join(args.save_dir, 'B2A_encodings.csv'), 'a') as f:
            B2A_encodings.to_csv(f, index=False, header=False)


if __name__ == "__main__":
    os.makedirs(args.save_dir, exist_ok = True)
    
    for epoch in range(1, args.epochs + 1):
        train(epoch, is_final = epoch == args.epochs)
        # TODO: build 'test' function
    print('encoding all')
    encode_all()
    
    
    
    

