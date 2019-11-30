import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset 
from torchvision import datasets, transforms

# TODO: inheret Dataset to include corruption method
#print('corrupting omics domain')
#n_samples_to_add = int(train_ome.shape[1] * args.added_noise /
#                       (1 - args.added_noise))
#train_shape_to_add = (train_ome.shape[0], n_samples_to_add)
#test_shape_to_add = (test_ome.shape[0], n_samples_to_add)
#sample_space = np.reshape(train_ome, -1)
#train_samples = np.random.choice(sample_space, 
#                                 np.prod(train_shape_to_add))
#test_samples = np.random.choice(sample_space, 
#                                np.prod(test_shape_to_add))
#train_samples = np.reshape(train_samples, train_shape_to_add)
#test_samples = np.reshape(test_samples, test_shape_to_add)
#train_ome = np.concatenate((train_ome, train_samples), axis = 1)
#test_ome = np.concatenate((test_ome, test_samples), axis = 1)

class XAEdataset(Dataset):
    '''Configure data loader for both domains
    Arguments:
        A_datafile (str): datafile to load to domain A
        B_datafile (str): datafile to load to domain B
        A_type (str): defines as either image or omics dataset
        B_type (str): defines as either image or omics dataset
        A_transforms (obj): pytorch composed transforms for A domain
        B_transforms (obj): pytorch composed transforms for B domain    
    Returns:
        A tuple of samples drawn from A and B
    '''
    def __init__(self, A_datafile, B_datafile, A_type, B_type, 
                 A_transforms=None, B_transforms=None):
        if A_type == 'img':
            self.A_data = np.load(A_datafile)
        elif A_type == 'ome':
            self.A_data = pd.read_csv(A_datafile).to_numpy()
        if B_type == 'img':
            self.B_data = np.load(B_datafile)
        elif B_type == 'ome':
            self.B_data = pd.read_csv(B_datafile).to_numpy()
        
        self.A_transforms = A_transforms
        self.B_transforms = B_transforms
        
    def __len__(self):
        return min(self.A_data.shape[0], self.B_data.shape[0])
    
    def data_dim(self):
        return self.A_data.shape[1:], self.B_data.shape[1:]
    
    def __getitem__(self, idx):
        A_batch = self.A_data[idx,:]
        B_batch = self.B_data[idx,:]
        if self.A_transforms is not None:
            A_batch = self.A_transforms(A_batch)
        if self.B_transforms is not None:
            B_batch = self.B_transforms(B_batch)
        return A_batch, B_batch

class MNISTdataset(Dataset):
    '''Configure data loader for MNIST example
    Arguments:
        save_dir: where to save data labels
    Returns:
        None
    '''
    def __init__(self, save_dir):
        self.save_dir = save_dir
        
        # load train and test
        train_mnist = datasets.MNIST('../data', 
                                     train = True, # set to False for test
                                     download = True,
                                     transform = transforms.ToTensor())
        
        self.A_data = train_mnist.train_data.unsqueeze(1).float()/255
        self.B_data = self.A_data.reshape((len(self.A_data), -1))
        # save labels
        train_labels = train_mnist.train_labels
        pd.DataFrame(train_labels.numpy()).to_csv(os.path.join(self.save_dir,
                    'train_labels.csv'),header=None, index=False)
        
    def __len__(self):
        return self.A_data.shape, self.B_data.shape
    
    def __getitem__(self, idx):
        A_batch = self.A_data[idx,:]
        B_batch = self.B_data[idx,:]
        return A_batch, B_batch
    
    