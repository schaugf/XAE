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
    '''Base class for an XAE dataset with data corruption method
    '''
    def corrupt_input(self, pct_noise):
        print('corrupting omics domain')
        dsize = 5000  # this should be whatever __len__() returns
        n_samples_to_add = int(dsize * pct_noise /
                               (1 - pct_noise))
        train_shape_to_add = (dsize, n_samples_to_add)
        sample_space = np.reshape(self.data, -1)
        train_samples = np.random.choice(sample_space, 
                                         np.prod(train_shape_to_add))
        train_samples = np.reshape(train_samples, train_shape_to_add)
        self.data = np.concatenate((self.data, train_samples), axis = 1)
    
#class OMEdataset(XAEdataset):  # if XAEdataset is implemented
class OMEdataset(Dataset):
    '''Configure data loader for both domains
    Arguments:
        datafile (str): datafile to load to domain A
        transforms (obj): pytorch composed transforms for A domain
    Returns:
        A tuple of samples drawn from A and B
    '''
    def __init__(self, datafile, transforms=None):
        self.data = pd.read_csv(datafile)
        self.features = [f for f in self.data.columns]
        self.data = self.data.to_numpy()
        self.transforms = transforms        
        
    def __len__(self):
        return len(self.data)
    
    def data_dim(self):
        return self.data.shape[1:]
    
    def __getitem__(self, idx):
        batch = self.data[idx,:]
        if self.transforms is not None:
            batch = self.transforms(batch)
        return batch

class IMGdataset(Dataset):
    '''Configure data loader for both domains
    Arguments:
        datafile (str): datafile to load to domain A
        transforms (obj): pytorch composed transforms for A domain
    Returns:
        A tuple of samples drawn from A and B
    '''
    def __init__(self, datafile, transforms=None):
        self.data = np.load(datafile)
        self.transforms = transforms        
        
    def __len__(self):
        return len(self.data)
    
    def data_dim(self):
        return self.data.shape[1:]
    
    def __getitem__(self, idx):
        batch = self.data[idx,:]
        if self.transforms is not None:
            batch = self.transforms(batch)
        return batch


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
    
    