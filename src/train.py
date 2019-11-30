import os
import sys
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms
from losses import vae_loss, vce_loss, mutual_encoding_loss
from data_loader import XAEdataset, MNISTdataset
from model import XAE

def train(epoch, is_final):
    '''Split datafiles into testing and training sets
    
    Arguments:
        save_dir (str): location to save training and testing sets
        processed_dir (str): location of processed slides
        slide_annot (str): annotation file for all slides
        split (float): fraction of data to be in training set
          
    Returns:
        A pair of tuples containing tile stacks and annotations for both
        training and test data sets
    '''
    model.train()
    loss_keys = ['A_vce_loss',
                 'B_vce_loss',
                 'A_vae_loss',
                 'B_vae_loss',
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
        
        # push data through model
        return_dict = model(A_data, B_data)
        
        # compute losses
        A_vce_loss = vce_loss(recon_x = return_dict['A2B2A_rec'], 
                              x = A_data, 
                              mu_1 = return_dict['A_mu'], 
                              logvar_1 = return_dict['A_logvar'], 
                              mu_2 = return_dict['A2B_mu'], 
                              logvar_2 = return_dict['A2B_logvar'])
        
        B_vce_loss = vce_loss(recon_x = return_dict['B2A2B_rec'], 
                              x = B_data, 
                              mu_1 = return_dict['B_mu'], 
                              logvar_1 = return_dict['B_logvar'], 
                              mu_2 = return_dict['B2A_mu'], 
                              logvar_2 = return_dict['B2A_logvar'])
        
        A_vae_loss = vae_loss(recon_x = return_dict['A_rec'], 
                              x = A_data, 
                              mu = return_dict['A_mu'], 
                              logvar = return_dict['A_logvar'])
        
        B_vae_loss = vae_loss(recon_x = return_dict['B_rec'], 
                              x = B_data, 
                              mu = return_dict['B_mu'], 
                              logvar = return_dict['B_logvar'])
        
        A_med_loss = mutual_encoding_loss(z1 = return_dict['A_z'], 
                                          z2 = return_dict['A2B_z'])
        
        B_med_loss = mutual_encoding_loss(z1 = return_dict['B_z'], 
                                          z2 = return_dict['B2A_z'])
        
        gate_norm_loss = torch.norm(model.B_encoder[0].weight)
        
        xae_loss = args.A_vce_lambda * A_vce_loss + \
                   args.B_vce_lambda * B_vce_loss + \
                   args.A_vae_lambda * A_vae_loss + \
                   args.B_vae_lambda * B_vae_loss + \
                   args.A_med_lambda * A_med_loss + \
                   args.B_med_lambda * B_med_loss + \
                   args.gate_norm_lambda * gate_norm_loss
        
        xae_loss.backward()
        optimizer.step()
        
        loss_vals = [A_vce_loss.item(),
                     B_vce_loss.item(),
                     A_vae_loss.item(),
                     B_vae_loss.item(),
                     A_med_loss.item(),
                     B_med_loss.item()]
        
        all_losses.append(dict(zip(loss_keys, loss_vals)))
            
    # at end of epoch
    print('Epoch:', epoch, 'XAE loss:', round(xae_loss.item(), 0))

    mean_loss = pd.DataFrame(pd.DataFrame(all_losses).mean()).T
    mean_loss['epoch'] = epoch
    
    with open(os.path.join(args.save_dir, 'xae_loss.csv'), 'a') as f:
        mean_loss.to_csv(f, 
                         header = [True if epoch == 1 else False][0], 
                         index = False)
    
    # save batch of reconstructions
    if (args.do_gate_layer) & (epoch % args.n_epoch_set_binary == 0):
        set_binary_layer(epoch)

    save_batch_reconstructions(return_dict, A_data, epoch)
    
    
def save_batch_reconstructions(return_dict, A_data, epoch):
    '''Split datafiles into testing and training sets
    
    Arguments:
        save_dir (str): location to save training and testing sets
        processed_dir (str): location of processed slides
        slide_annot (str): annotation file for all slides
        split (float): fraction of data to be in training set
          
    Returns:
        A pair of tuples containing tile stacks and annotations for both
        training and test data sets
    '''
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
    
    to_save = Image.fromarray((255*imgs_to_save).astype(np.uint8))
    to_save.save(os.path.join(args.save_dir, 'image_reconstructions.png'))
    

def set_binary_layer(epoch):
    '''Set the binary layer on the gate layer
    Arguments:
        epoch (int): training epoch
    Returns:
        None
    '''
    print('setting binary layer')
    gate_weights = model.B_encoder[0].weight.detach().cpu()
    gwpd = pd.DataFrame({'gate_weights':gate_weights.numpy()[0]})
    is_noise = np.ones(gwpd.shape[0])
    is_noise[:B_init_dim[0]] = 0
    gwpd['is_noise'] = is_noise
    noise_weight = gwpd['gate_weights'][gwpd['is_noise']==1]
    # compute quantile for noise weights
    noise_quantile_cutoff = (noise_weight**2).quantile(args.quantile_cutoff)
    # set binary layer
    binary_layer_set = (gwpd['gate_weights']**2) > noise_quantile_cutoff
    binary_layer_set = torch.tensor(binary_layer_set.astype(float))
    if args.cuda:
        binary_layer_set = nn.Parameter(binary_layer_set.cuda())
    else:
        binary_layer_set = nn.Parameter(binary_layer_set)
    model.B_encoder[0].binary_layer = binary_layer_set
    save_gate_weights(epoch)
    print('weights set')
    
def save_gate_weights(epoch):
    '''Split datafiles into testing and training sets
    
    Arguments:
        save_dir (str): location to save training and testing sets
        processed_dir (str): location of processed slides
        slide_annot (str): annotation file for all slides
        split (float): fraction of data to be in training set
          
    Returns:
        A pair of tuples containing tile stacks and annotations for both
        training and test data sets
    '''
    gate_weights = model.B_encoder[0].weight
    gate_binary = model.B_encoder[0].binary_layer
    
    gw = pd.DataFrame({'gate_weights': gate_weights.detach().cpu()[0].numpy().T,
                       'gate_binary' : gate_binary.detach().cpu().numpy().T})
    
    gw.to_csv(os.path.join(args.save_dir, 'gate_weights.csv'), index=False)
    
    # remove elements to make square matrix
    nrm = np.mod(gw.shape[0], train_img.shape[2])
    if nrm !=0 :
        gw = gw.iloc[:-nrm,]
    
    # TODO: check if image exists in filesystem, load else create
    
    # TODO: load image, append as in reconstruction
    g = np.array(gw['gate_weights']).reshape(-1, train_img.shape[2])
    g = (255*((g**2)/(g**2).max())).astype(np.uint8)
    Image.fromarray(g).save(os.path.join(args.save_dir, 'gate_2d.jpg'))
    
    # binary weights
    b = np.array(gw['gate_binary']).reshape(-1, train_img.shape[2])
    b = (b*255).astype(np.uint8)
    Image.fromarray(b).save(os.path.join(args.save_dir, 'gate_binary_2d.jpg'))

    #Image.fromarray(g).show()
    #Image.fromarray(b).show()
    
    
def encode_all(loader):
    '''Encode entire dataset and save to file
    Arguments:
        loader (torch): pytorch dataloader object
    Returns:
        None
    '''
    A_imgs = []
    A_predicted_imgs = []
    for batch_idx, (A_data, B_data) in enumerate(loader):
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
    # save a bunch of data
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
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train XAE Model')
    parser.add_argument('--A_datafile', type=str, 
                        default='../data/CEDAR_prostate/scATAC_xae.csv',
                        help='pointer to A domain datafile (csv or npy)')
    parser.add_argument('--B_datafile', type=str, 
                        default='../data/CEDAR_prostate/cycIF_xae.csv',
                        help='pointer to A domain datafile (csv or npy)')
    parser.add_argument('--A_type', type=str, default='ome',
                        help='A domain type (img or ome)')
    parser.add_argument('--B_type', type=str, default='ome',
                        help='B domain type (img or ome)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size for training (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='optimizer learning rate (default: 1e-3)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--save_dir', type=str, default='../results/test', 
                        help='save directory')
    parser.add_argument('--dataset', type=str, default='test',
                        help='dataset to load (mnist or test)')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='size of learned latent dimension')
    parser.add_argument('--inter_dim', type=int, default=128,
                        help='how big is linear around z')
    parser.add_argument('--added_noise', type=float, default=0.5,
                        help='fraction of noise added to input')
    parser.add_argument('--do_gate_layer', type=int, default=1,
                        help='append weighted gate layer to input?')
    parser.add_argument('--A_med_lambda', type=float, default=10.0,
                        help='A mes loss coefficient')
    parser.add_argument('--B_med_lambda', type=float, default=10.0,
                        help='B med loss coefficient')
    parser.add_argument('--A_vce_lambda', type=float, default=3.0,
                        help='A vce loss coefficient')
    parser.add_argument('--B_vce_lambda', type=float, default=3.0,
                        help='B vce loss coefficient')
    parser.add_argument('--A_vae_lambda', type=float, default=1.0,
                        help='A vae loss coefficient')
    parser.add_argument('--B_vae_lambda', type=float, default=1.0,
                        help='B vae loss coefficient')
    parser.add_argument('--gate_norm_lambda', type=int, default=1,
                        help='lambda penalty on gate norm')
    parser.add_argument('--n_epoch_set_binary', type=int, default=25,
                        help='how often to set binary gate layer')
    parser.add_argument('--quantile_cutoff', type=float, default=0.5,
                        help='quantile of noise weights to set to zero')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    
    args.cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': False} if args.cuda =='cuda' else {}
    os.makedirs(args.save_dir, exist_ok = True)
    
    # TODO: confugre domain-specific transforms
    img_transforms = transforms.Compose([transforms.ToPILImage(),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ColorJitter(brightness = 0.2,
                                                       contrast = 0.2,
                                                       saturation = 0.2,
                                                       hue = 0.2),
                                transforms.ToTensor()])
    ome_transforms = None
                                
    # Configure datasets
    if args.dataset == 'MNIST':
        train_dataset = MNISTdataset(save_dir = args.save_dir)
    elif ((args.A_datafile is not None) & (args.B_datafile is not None)):
        train_dataset = XAEdataset(A_datafile = args.A_datafile, 
                                   B_datafile = args.B_datafile,
                                   A_type = args.A_type,
                                   B_type = args.B_type,
                                   A_transforms = None, 
                                   B_transforms = None)
    else:
        sys.exit('both datafiles are required')
    
    # define initial dimensions
    A_shape, B_shape = train_dataset.data_dim()

    # Configure data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               shuffle = True,
                                               num_workers = 4, 
                                               pin_memory = False)    
                     
    model = XAE(A_type = 'ome', 
                B_type = 'ome',
                A_shape = A_shape,
                B_shape = B_shape)

    if args.cuda:
        model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr = args.lr)

    for epoch in range(1, args.epochs + 1):
        print('beginning epoch', epoch)
        train(epoch, is_final = epoch == args.epochs)
        
    #print('encoding all')
    #encode_all(eval_loader)
    
