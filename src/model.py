import numpy as np
import torch
from torch import nn
from custom_layers import Flatten, Rachet, GateLayer

class XAE(nn.Module):    
    '''XAE model for jointly learned latent representation
    Arguments:
        A_type (str): define domain A as either 'img' or 'ome'
        B_type (str): define domain B as either 'img' or 'ome'
        A_shape (tuple): shape of A domain input
        B_shape (tuple): shape of B domain input
        latent_dim (int): dimension of learned latent representation
        inter_dim (int): dimension of intermediate dimension
        do_gate_A (bool): gate input layer for A
        do_gate_B (bool): gate input layer for B
    Returns:
        Pytorch model of XAE learning architecture
    '''
    def __init__(self, 
                 A_type = 'img', 
                 B_type = 'ome',
                 A_shape = (),
                 B_shape = (),
                 latent_dim = 8,
                 inter_dim = 32,
                 do_gate_A = False,
                 do_gate_B = False):
        
        super(XAE, self).__init__()
        self.A_type = A_type
        self.B_type = B_type
        self.A_shape = A_shape
        self.B_shape = B_shape
        self.latent_dim = latent_dim
        self.inter_dim = inter_dim
        self.do_gate_A = do_gate_A
        self.do_gate_B = do_gate_B
        
        # shared latent feature layers
        self.shared_fc1 = nn.Linear(self.inter_dim, self.latent_dim)
        self.shared_fc2 = nn.Linear(self.inter_dim, self.latent_dim)
        
        # configure A-domain networks
        if self.A_type == 'img':
            self.A_encoder = self.make_image_encoder(in_shape = A_shape)
            self.A_decoder = self.make_image_decoder(out_shape = A_shape)
        elif self.A_type == 'ome':
            self.A_encoder = self.make_omic_encoder(in_shape = A_shape, 
                                                    do_gate = self.do_gate_A)
            self.A_decoder = self.make_omic_decoder(out_shape = A_shape)
        
        # configure B-domain networks
        if self.B_type == 'img':
            self.B_encoder = self.make_image_encoder(in_shape = B_shape)
            self.B_decoder = self.make_image_decoder(out_shape = B_shape)
        elif self.B_type == 'ome':
            self.B_encoder = self.make_omic_encoder(in_shape = B_shape,
                                                    do_gate = self.do_gate_B)
            self.B_decoder = self.make_omic_decoder(out_shape = B_shape)
    
    def make_image_encoder(self, in_shape):
        '''Configure image encoder
        Arguments:
            in_shape (tuple): shape of domain input
        Returns:
            Image encoder architecture
        '''
        return nn.Sequential(
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
    
    def make_image_decoder(self, out_shape):
        '''Configure image decoder
        Arguments:
            out_shape (tuple): shape of domain output
        Returns:
            Image decoder architecture
        '''
        return nn.Sequential(
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
    
    def make_omic_encoder(self, in_shape, do_gate):
        '''Configure omic encoder
        Arguments:
            in_shape (tuple): input tensor shape
        Returns:
            Omic encoder network
        '''
        if do_gate:
            print('gating input')
            return nn.Sequential(
                GateLayer(in_shape[0]),
                nn.Tanh(),
                nn.Linear(in_shape[0], self.inter_dim*8),
                nn.ReLU(),
                nn.Linear(self.inter_dim*8, self.inter_dim*4),
                nn.ReLU(),
                nn.Linear(self.inter_dim*4, self.inter_dim),
                nn.ReLU()
                )
        else:
            return nn.Sequential(
                nn.Linear(in_shape[0], self.inter_dim*8),
                nn.ReLU(),
                nn.Linear(self.inter_dim*8, self.inter_dim*4),
                nn.ReLU(),
                nn.Linear(self.inter_dim*4, self.inter_dim),
                nn.ReLU()
                )    
        
    def make_omic_decoder(self, out_shape):
        '''Configure omic decoder network
        Arguments:
            out_shape (tuple): output shape of omic network
        Returns:
            Omic decoder network
        '''
        return nn.Sequential(
                nn.Linear(self.latent_dim, self.inter_dim),
                nn.ReLU(),
                nn.Linear(self.inter_dim, self.inter_dim*8),
                nn.ReLU(),
                nn.Linear(self.inter_dim*8, out_shape[0]),
                nn.ReLU()
                )
        
    def reparameterize(self, mu, logvar):
        '''VAE reparameterization trick
        Arguments:
            mu (float): latent vector means
            logvar (float): latent vector log variance
        Returns:
            reparameterized representation of latent distribution
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + (eps * std)

    def forward(self, A_in, B_in):
        '''Forward pass over the XAE network
        Arguments:
            A_in (tensor): data from domain A
            B_in (tensor): data from domain B
        Returns:
            A_rec (tensor): A autoencoder reconstruction
            B_rec (tensor): B autoencoder reconstruction
            A_mu (tensor): Latent means of encoded A
            A_logvar (tensor): Latent log variance of encoded A
            A_z (tensor): Latent z of encoded A
            B_mu (tensor): Latent means of encoded B
            B_logvar (tensor): Latent log variance of encoded B
            B_z (tensor): Latent z of encoded B
            A2B_pred (tensor): Cross-domain predictions from A to B
            B2A_pred (tensosr): Cross-domain predictions from B to A
            A2B_mu (tensor): Latent means from A to B
            A2B_logvar (tensor): Latent log variance from A to B
            A2B_z (tensor): Latent representation from A to B
            B2A_mu (tensor): Latent means from B to A
            B2A_logvar (tensor): Latent log variance from B to A
            B2A_z (tensor): Latent representation from B to A
            A2B2A_rec (tensor): A reconstruction through B
            B2A2B_rec (tensor): B reconstruction through A
        '''
        A_inter = self.A_encoder(A_in)
        A_mu = self.shared_fc1(A_inter)
        A_logvar = self.shared_fc2(A_inter)
        A_z = self.reparameterize(A_mu, A_logvar)
        A_rec = self.A_decoder(A_z)
        
        B_inter = self.B_encoder(B_in)
        B_mu = self.shared_fc1(B_inter) 
        B_logvar = self.shared_fc2(B_inter)
        B_z = self.reparameterize(B_mu, B_logvar)
        B_rec = self.B_decoder(B_z)
        
        # cross-domain generators
        A2B_pred = self.B_decoder(A_z)
        B2A_pred = self.A_decoder(B_z)
        
        # cycle autoencoders
        A2B_inter = self.B_encoder(A2B_pred)
        A2B_mu = self.shared_fc1(A2B_inter)
        A2B_logvar = self.shared_fc2(A2B_inter)
        A2B_z = self.reparameterize(A2B_mu, A2B_logvar)
        
        B2A_inter = self.A_encoder(B2A_pred)
        B2A_mu = self.shared_fc1(B2A_inter)
        B2A_logvar = self.shared_fc2(B2A_inter)
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
        