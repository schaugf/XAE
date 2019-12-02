import numpy as np
import torch
from torch.nn import functional as F

def vae_loss(x, recon_x, mu, logvar):
    '''Variational autoencoding loss
    Arguments:
        x (tensor): input 
        recon_x (tensor): reconstructed copy of x
        mu (tensor): latent distribution means
        logvar (tensor): latent distribution log variances
    Returns:
        Autoencoding loss plus KL divergence
    '''
    BCE = F.binary_cross_entropy(recon_x.view(-1, np.prod(recon_x.shape[1:])),
                                 x.view(-1, np.prod(x.shape[1:])), 
                                 reduction = 'sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + (0.1 * KLD)

def vce_loss(x, recon_x, mu_1, logvar_1, mu_2, logvar_2):
    '''Variational cycle consistency loss
    Arguments:
        x (tensor): input data
        recon_x (tensor): reconstructed x
        mu_1 (tensor): latent dimension means from first encoding
        logvar_1 (tensor): latent dimension log variance from first encoding
        mu_2 (tensor): latent dimension means from second encoding
        logvar_2 (tensor): latent dimension log variance from second encoding
    Returns:
        Loss term including both first and second divergence terms
    '''
    CC = F.binary_cross_entropy(recon_x.view(-1, np.prod(recon_x.shape[1:])), 
                                x.view(-1, np.prod(x.shape[1:])),
                                reduction = 'sum')
    KLD_1 = -0.5 * torch.sum(1 + logvar_1 - mu_1.pow(2) - logvar_1.exp())
    KLD_2 = -0.5 * torch.sum(1 + logvar_2 - mu_2.pow(2) - logvar_2.exp())
    return CC + (0.05 * KLD_1) + (0.05 * KLD_2)

def mutual_encoding_loss(z1, z2):
    '''Mutual encoding loss term
    Arguments:
        z1 (tensor): latent representation encoding from source domain
        z2 (tensor): latent representation encoding from target domain
    Returns:
        Mutual encoding loss ternsor
    '''
    return F.mse_loss(z1, z2) * z1.shape[0]
    