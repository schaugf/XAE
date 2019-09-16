from __future__ import print_function
import os
import argparse
from random import shuffle
from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import init
from torchvision import datasets, transforms
#from torchsummary import summary


parser = argparse.ArgumentParser(description='XAE Model')
parser.add_argument('--batch_size', type=int, default=12, metavar='N',
                    help='input batch size for training (default: 12)')
parser.add_argument('--added_noise', type=float, default=0.2, metavar='N',
                    help='fraction of noise added to input')
parser.add_argument('--do_gate_layer', type=int, default=1, metavar='N',
                    help='append weighted gate layer to input?')
parser.add_argument('--do_gate_recon', type=int, default=1, metavar='N',
                    help='gate reconstruction?')
parser.add_argument('--gate_recon_lambda', type=int, default=250, metavar='N',
                    help='weight gate reconstruction term')
parser.add_argument('--do_vae_loss', type=int, default=1, metavar='N',
                    help='include vae loss term')
parser.add_argument('--A_vae_lambda', type=float, default=1.0, metavar='N',
                    help='A vae loss coefficient')
parser.add_argument('--A_cycle_lambda', type=float, default=1.0, metavar='N',
                    help='A cycle loss coefficient')
parser.add_argument('--A_med_lambda', type=float, default=1.0, metavar='N',
                    help='A mes loss coefficient')

parser.add_argument('--B_vae_lambda', type=float, default=1.0, metavar='N',
                    help='B vae loss coefficient')
parser.add_argument('--B_cycle_lambda', type=float, default=1.0, metavar='N',
                    help='B cycle loss coefficient')
parser.add_argument('--B_med_lambda', type=float, default=1.0, metavar='N',
                    help='B med loss coefficient')

parser.add_argument('--n_epoch_set_binary', type=int, default=20, metavar='N',
                    help='how often to set binary gate layer')
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
os.makedirs(args.save_dir, exist_ok = True)


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
train_labels = train_mnist.train_labels

# save labels
pd.DataFrame(train_labels.numpy()).to_csv(os.path.join(args.save_dir,
            'train_labels.csv'),header=None, index=False)

test_img = test_mnist.test_data.unsqueeze(1).float()/255
test_ome = test_img.reshape((len(test_img), -1))

if args.dataset == 'test':
    train_img = train_img[:200]
    train_ome = train_ome[:200]
    test_img = test_img[:200]
    test_ome = test_ome[:200]

# define initial dimensions
A_type = 'img'
B_type = 'ome'
A_init_dim = train_img.shape[1:]
B_init_dim = train_ome.shape[1:]

print('corrupting omics domain')
n_samples_to_add = int(train_ome.shape[1] * 
                       args.added_noise /
                       (1 - args.added_noise))

train_shape_to_add = (train_ome.shape[0], n_samples_to_add)
test_shape_to_add = (test_ome.shape[0], n_samples_to_add)

sample_space = np.reshape(train_ome, -1)

train_samples = np.random.choice(sample_space, 
                                 np.prod(train_shape_to_add))

test_samples = np.random.choice(sample_space, 
                                np.prod(test_shape_to_add))

train_samples = np.reshape(train_samples, train_shape_to_add)
test_samples = np.reshape(test_samples, test_shape_to_add)

train_ome = np.concatenate((train_ome, train_samples), axis = 1)

test_ome = np.concatenate((test_ome, test_samples), axis = 1)

print('corrupted omic train shape', train_ome.shape)
print('corrupted omic test shape', test_ome.shape)
        
# define datasets
train = torch.utils.data.TensorDataset(torch.tensor(train_img).float(), 
                                       torch.tensor(train_ome).float())

eval_set = torch.utils.data.TensorDataset(torch.tensor(train_img).float(), 
                                          torch.tensor(train_ome).float())

test = torch.utils.data.TensorDataset(torch.tensor(test_img).float(), 
                                      torch.tensor(test_ome).float())

# set data parameters
A_shape = train_img.shape[1:]
B_shape = train_ome.shape[1:]

print('A_shape:', A_shape, 'B_shape', B_shape)

# Define data loaders
train_loader =  torch.utils.data.DataLoader(train, 
                                            batch_size = args.batch_size, 
                                            shuffle = True)

eval_loader = torch.utils.data.DataLoader(eval_set, 
                                          batch_size = args.batch_size, 
                                          shuffle = False)

test_loader =  torch.utils.data.DataLoader(test, 
                                           batch_size = args.batch_size, 
                                           shuffle = False)


class Flatten(nn.Module):
        def __init__(self):
            super(Flatten, self).__init__()
            
        def forward(self, x):
            return x.view(x.size()[0], -1)
        
        
class Rachet(nn.Module):
    def __init__(self, out_shape):
        super(Rachet, self).__init__()
        self.out_shape = out_shape
        
    def forward(self, x):
        return x.view(x.size(0), 8, self.out_shape[1], self.out_shape[2])
    

class GateLayer(nn.Module):
    def __init__(self, in_shape):
        super(GateLayer, self).__init__()
        self.in_shape = in_shape
        self.weight = nn.Parameter(torch.Tensor(1, in_shape))
        init.kaiming_uniform_(self.weight)

    def forward(self, x):
        return x * self.weight
    
    
class BinaryGate(nn.Module):
    # TODO: set non-trainable
    def __init__(self, in_shape):
        super(BinaryGate, self).__init__()
        self.binary_layer = torch.tensor(np.ones(in_shape))
    
    def forward(self, x):
        return x * self.binary_layer
    

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
                Flatten(),
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
                Rachet(out_shape),
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
        
        if args.do_gate_layer:
            print('gating input')
            ome_encoder = nn.Sequential(
                GateLayer(in_shape[0]),
                nn.Tanh(),
                nn.Linear(in_shape[0], self.inter_dim*8),
                nn.ReLU(),
                nn.Linear(self.inter_dim*8, self.inter_dim),
                nn.ReLU()
                )
        else:
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


def vce_loss(recon_x, x, mu, logvar):
    # variational cyclic encoder
    # cycle consistency loss with KL divergence
    CC = F.binary_cross_entropy(recon_x.view(-1, np.prod(recon_x.shape[1:])), 
                                x.view(-1, np.prod(x.shape[1:])),
                                reduction = 'sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return CC + KLD
    

def mutual_encoding_loss(z1, z2):
    # encoding divergence between omics and imaging domains
    MED = F.mse_loss(z1, z2) * len(z1)
    return MED
    

def train(epoch, is_final):
    model.train()
    # TODO: define this in main loop
    loss_keys = ['A_vae_loss',
                 'B_vae_loss',
                 'A_cycle_loss',
                 'B_cycle_loss',
                 'A_med_loss',
                 'B_med_loss',
                 'total_loss']
    all_losses = []

    for batch_idx, (A_data, B_data) in enumerate(train_loader):
        #A_data, B_data = next(iter(train_loader))  # for debugging
        A_data = Variable(A_data)
        B_data = Variable(B_data)
        
        if args.cuda:
            A_data = A_data.cuda()
            B_data = B_data.cuda()
            
        optimizer.zero_grad()
        
        return_dict = model(A_data, B_data)
        
        # compute losses
        # TODO: set "do_vae_loss" flag
        A_vae_loss = vae_loss(recon_x = return_dict['A_rec'], 
                              x = A_data, 
                              mu = return_dict['A_mu'], 
                              logvar = return_dict['A_logvar'])
        
        A_cycle_loss = cycle_loss(recon_x = return_dict['A2B2A_rec'],
                                  x = A_data)
        
        if args.do_gate_recon:
            Wtanh2 = model.B_encoder[0].weight.detach().tanh()**2
            
            B_cycle_loss = cycle_loss(recon_x = return_dict['B2A2B_rec'] * Wtanh2,
                                      x = B_data * Wtanh2)
            
            B_vae_loss = vae_loss(recon_x = return_dict['B_rec'] * Wtanh2, 
                              x = B_data * Wtanh2, 
                              mu = return_dict['B_mu'], 
                              logvar = return_dict['B_logvar'])
            
            # weight gated recon terms
            B_cycle_loss = B_cycle_loss * args.gate_recon_lambda
            B_vae_loss = B_vae_loss * args.gate_recon_lambda
            
        else:
           B_cycle_loss = cycle_loss(recon_x = return_dict['B2A2B_rec'],
                                      x = B_data)
           
           B_vae_loss = vae_loss(recon_x = return_dict['B_rec'], 
                  x = B_data, 
                  mu = return_dict['B_mu'], 
                  logvar = return_dict['B_logvar'])
        
        A_med_loss = mutual_encoding_loss(z1 = return_dict['A_z'], 
                                          z2 = return_dict['A2B_z'])
        
        B_med_loss = mutual_encoding_loss(z1 = return_dict['B_z'], 
                                          z2 = return_dict['B2A_z'])
        
        # check do_vae_loss
        xae_loss = args.A_vae_lambda * A_vae_loss + \
                   args.B_vae_lambda * B_vae_loss + \
                   args.A_cycle_lambda * A_cycle_loss + \
                   args.B_cycle_lambda * B_cycle_loss + \
                   args.A_med_lambda * A_med_loss + \
                   args.B_med_lambda * B_med_loss
                   
        xae_loss.backward()
        optimizer.step()
        
        # TODO: if do_vae_loss, remove from storage
        loss_vals = [A_vae_loss.item(),
                     B_vae_loss.item(),
                     A_cycle_loss.item(),
                     B_cycle_loss.item(),
                     A_med_loss.item(),
                     B_med_loss.item()]
        
        all_losses.append(dict(zip(loss_keys, loss_vals)))
            
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
    nrec = 8
    gt_imgs = A_data[:nrec,0,:].detach().cpu().numpy()
    gt_stack = np.concatenate(list(gt_imgs))
    
    img_rec = return_dict['A_rec'][:nrec,0,:].detach().cpu().numpy()
    rec_stack = np.concatenate(list(img_rec))
    
    img_cyc_rec = return_dict['A2B2A_rec'][:nrec,0,:].detach().cpu().numpy()
    cyc_rec_stack = np.concatenate(list(img_cyc_rec))
    
    # filter out added noise for reconstructions
    A2B = return_dict['A2B_pred'][:nrec, :B_init_dim[0]]
    A2B_stack = np.concatenate(list(A2B.reshape(nrec, 28, 28).detach().cpu().numpy()))
    
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
    
    # TODO: set binary layer
    if epoch % args.n_epoch_set_binary:
        gate_weights = model.B_encoder[0].weight.detach().cpu()
        gwpd = pd.DataFrame({'gate_weights':gate_weights.numpy()[0]})
        gwpd['gate_weights_squared'] = gwpd['gate_weights'] ** 2 
        gwpd['gate_weights_tanh'] = np.tanh(gwpd['gate_weights'])
        #gwpd['is_noise']
        # compute .95 quantile for noise weights
        # set binary layer
        
    
    to_save = Image.fromarray((255*imgs_to_save).astype(np.uint8))
    to_save.save(os.path.join(args.save_dir, 'image_reconstructions.png'))
        

def encode_all():
    A_imgs = []
    A_predicted_imgs = []
    for batch_idx, (A_data, B_data) in enumerate(eval_loader):
        A_data = Variable(A_data)
        B_data = Variable(B_data)
        
        if args.cuda:
            A_data = A_data.cuda()
            B_data = B_data.cuda()
            
        return_dict = model(A_data, B_data)
    
        # save encodings
        A_encodings = pd.DataFrame(return_dict['A_z'].detach().cpu().numpy())
        with open(os.path.join(args.save_dir, 'A_encodings.csv'), 'a') as f:
            A_encodings.to_csv(f, index=False, header=False)
        
        B_encodings = pd.DataFrame(return_dict['B_z'].detach().cpu().numpy())
        with open(os.path.join(args.save_dir, 'B_encodings.csv'), 'a') as f:
            B_encodings.to_csv(f, index=False, header=False)
        
        # save cycle-encodings
        A2B_encodings = pd.DataFrame(return_dict['A2B_z'].detach().cpu().numpy())
        with open(os.path.join(args.save_dir, 'A2B_encodings.csv'), 'a') as f:
            A2B_encodings.to_csv(f, index=False, header=False)
        
        B2A_encodings = pd.DataFrame(return_dict['B2A_z'].detach().cpu().numpy())
        with open(os.path.join(args.save_dir, 'B2A_encodings.csv'), 'a') as f:
            B2A_encodings.to_csv(f, index=False, header=False)
        
        # save actual omics data to csv
        B_data_save = pd.DataFrame(B_data.detach().cpu().numpy())
        with open(os.path.join(args.save_dir, 'B_data.csv'), 'a') as f:
            B_data_save.to_csv(f, index=False, header=False)
        
        # store actual images to nparray
        A_imgs.append(A_data.detach().cpu().numpy())
        
        # save predicted imaging-omics as csv
        A2B_pred = pd.DataFrame(return_dict['A2B_pred'].detach().cpu().numpy())
        with open(os.path.join(args.save_dir, 'A2B_prediction.csv'), 'a') as f:
            A2B_pred.to_csv(f, index=False, header=False)
        
        # store predicted omics-images as numpy arrays
        B2A_pred = return_dict['B2A_pred'].detach().cpu().numpy()
        A_predicted_imgs.append(B2A_pred)
        
        
    # save numpy image arrays
    np.save(os.path.join(args.save_dir, 'A_data.npy'), np.concatenate(A_imgs))
    np.save(os.path.join(args.save_dir, 'A_pred.npy'), np.concatenate(A_predicted_imgs))
    
    # Compute xent of image-to-omics translation, read both true omics and predicted omics
    B_data_full = torch.tensor(np.array(pd.read_csv(os.path.join(args.save_dir, 
                                                                 'B_data.csv'),
                                         header=None)))
    A2B_pred_full = torch.tensor(np.array(pd.read_csv(os.path.join(args.save_dir, 
                                                                   'A2B_prediction.csv'),
                                         header=None)))
    
    A2B_xents = [F.binary_cross_entropy(A2B_pred_full[i], B_data_full[i]).item() \
                 for i in range(len(A2B_pred_full))]
    
    pd.DataFrame(A2B_xents).to_csv(os.path.join(args.save_dir, 'A2B_xent.csv'),
                 header=None, index=False)
    
    # compute xent of omics-to-image translation, read both A_data and A_pred
    A_data_full = torch.tensor(np.concatenate(A_imgs))
    A_data_pred = torch.tensor(np.concatenate(A_predicted_imgs))
    
    B2A_xents = [F.binary_cross_entropy(A_data_pred[i], A_data_full[i]).item() \
                 for i in range(len(A_data_pred))]
    
    pd.DataFrame(B2A_xents).to_csv(os.path.join(args.save_dir, 'B2A_xent.csv'),
                 header=None, index=False)    
    
    # save gate weights
    if args.do_gate_layer:
        gate_weights = model.B_encoder[0].weight
        pd.DataFrame(gate_weights.detach().cpu().numpy()).T.to_csv(
                os.path.join(args.save_dir, 
                             'gate_weights.csv'))
        
        # save gate weight image
        g = gate_weights.detach().cpu().numpy()[0]
        nrm = np.mod(len(g), train_img.shape[2])
        if nrm != 0:
            g = g[:-nrm]
        g = g.reshape(-1, train_img.shape[2])
        g = g / g.max()
        g = g ** 2
        mx = (g*255).astype(np.uint8)
        
        Image.fromarray(mx).save(os.path.join(args.save_dir, 'gate_2d.jpg'))
        #Image.fromarray(mx).show()

        
if __name__ == "__main__":    
    for epoch in range(1, args.epochs + 1):
        shuffle(train_img)
        shuffle(train_ome)
        train_t = torch.utils.data.TensorDataset(torch.tensor(train_img).float(), 
                                                 torch.tensor(train_ome).float())
        # Define data loaders
        train_loader =  torch.utils.data.DataLoader(train_t, 
                                                batch_size = args.batch_size, 
                                                shuffle = True)

        train(epoch, is_final = epoch == args.epochs)
        # TODO: program binary layer
        # TODO: test w/ and w/o vae loss
        # TODO: build test function  
        
    print('encoding all')
    encode_all()
    
