import os
import numpy as np
import pandas as pd
from PIL import Image
import torch

def save_batch_reconstructions(return_dict, A_data, epoch, save_dir):
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
        tmp_img = Image.open(os.path.join(save_dir, 
                                          'image_reconstructions.png'))
        tmp_img = np.array(tmp_img) / 255
        imgs_to_save = np.concatenate((tmp_img,
                                       gt_stack,
                                       rec_stack,
                                       A2B_stack,
                                       cyc_rec_stack,
                                       np.ones((gt_stack.shape[0], 1))), axis=1)
    
    to_save = Image.fromarray((255*imgs_to_save).astype(np.uint8))
    to_save.save(os.path.join(save_dir, 'image_reconstructions.png'))
    

def compute_xent(save_dir):
    # compute xent of translation
    B_data_full = torch.tensor(np.array(pd.read_csv(os.path.join(save_dir, 
                                                                 'B_data.csv'),
                                         header=None)))
    A2B_pred_full = torch.tensor(np.array(pd.read_csv(os.path.join(save_dir, 
                                                                   'A2B_prediction.csv'),
                                         header=None)))
    A2B_xents = [F.binary_cross_entropy(A2B_pred_full[i], 
                                        B_data_full[i]).item() \
                 for i in range(len(A2B_pred_full))]
    
    pd.DataFrame(A2B_xents).to_csv(os.path.join(save_dir, 'A2B_xent.csv'),
                 header=None, index=False)
    
    A_data_full = torch.tensor(np.concatenate(A_imgs))
    A_data_pred = torch.tensor(np.concatenate(A_predicted_imgs))
    B2A_xents = [F.binary_cross_entropy(A_data_pred[i], 
                                        A_data_full[i]).item() \
                 for i in range(len(A_data_pred))]
    pd.DataFrame(B2A_xents).to_csv(os.path.join(save_dir, 'B2A_xent.csv'),
                 header=None, index=False)
    