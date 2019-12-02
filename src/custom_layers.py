import os
import numpy as np
import pandas as pd
import torch
from torch import nn

class Flatten(nn.Module):
    '''Custom layer to flatten into 1D vector
    Arguments:
        None
    Returns:
        Flattened input tensor
    '''
    def __init__(self):
        super(Flatten, self).__init__()    
    def forward(self, x):
        return x.view(x.size()[0], -1)
    
class Rachet(nn.Module):
    '''Rearranges input vector into output_shape
    Arguments:
        output_shape (tuple): desired output shape
    Returns:
        racheted view as input tensor
    '''
    def __init__(self, out_shape):
        super(Rachet, self).__init__()
        self.out_shape = out_shape
        
    def forward(self, x):
        return x.view(x.size(0), 8, self.out_shape[1], self.out_shape[2])
    
class GateLayer(nn.Module):
    '''Custom element-wise gate layer 
    Arguments:
        in_shape (tuple): shape of input tensor
    Returns:
        element-wise gate transform
    '''
    def __init__(self, in_shape):
        super(GateLayer, self).__init__()
        self.in_shape = in_shape
        self.weight = nn.Parameter(torch.Tensor(1, in_shape))
        nn.init.kaiming_uniform_(self.weight)
        self.binary_layer = nn.Parameter(torch.FloatTensor(np.ones(in_shape)))
        self.binary_layer.requires_grad = False
        self.gate_weights = self.weight[0].detach().cpu().numpy()

    def set_binary_layer(self, qcutoff):
        '''Set binary layer of the gate given a quantile cutoff
        Arguments:
            None
        Returns:
            None
        '''
        qthres = np.quantile(self.gate_weights**2, qcutoff)
        self.binary_layer[self.gate_weights**2 < qthres] = 0
         
    def save_gate_weights(self, save_dir, label=''):
        '''Save the gate weights to a csv file
        Arguments:
            save_dir (str): location to save gate weights
            label (str): label to append to file to make unique
        Returns:
            None
        '''
        gws = pd.DataFrame({'gate_weights': self.gate_weights})
        gws.to_csv(os.path.join(save_dir, label+'_gate_weights.csv'), 
                   index=False)

    def forward(self, x):
        return x * self.weight * self.binary_layer
    