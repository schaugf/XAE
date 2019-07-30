'''
XAE: Cycle Consistent Cross-Domain Autoencoder Architecture
'''

import os
import time
import argparse

from keras.models import Model
from keras.layers import Input, Dense, Lambda, Reshape, Flatten
from keras.layers import Conv2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.datasets import mnist

import numpy as np
import pandas as pd
from PIL import Image

os.environ['HDF5_USE_FILE_LOCKING']='FALSE' 


class XAE():
    ''' Instantiate a cycle consistent cross-domain autoencoder '''
    
    def __init__(self, 
                 learning_rate = 2e-4,
                 lambda_1 = 10.0,
                 lambda_2 = 10.0,
                 beta_1 = 0.5,
                 beta_2 = 0.99, 
                 latent_dim = 8, 
                 inter_dim = 64,
                 data_dir = 'data/test',
                 save_dir = '../results/test',
                 epochs = 2,
                 batch_size = 32,
                 do_save_model = False,
                 n_imgs_to_save = 30
                 ):
        
        # instantiate self parameters
        
        self.lr = learning_rate
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.latent_dim = latent_dim
        self.inter_dim = inter_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_imgs_to_save = n_imgs_to_save
        
        self.data_dir = data_dir
        self.date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        #self.save_dir = os.path.join('results', self.date_time)
        self.save_dir = save_dir
        
        self.do_save_model = do_save_model
        
        os.makedirs(self.save_dir, exist_ok = True)
        

        # load data
        
        self.LoadTestData()
        
        # configure optimizer
        
        self.optimizer = Adam(self.lr, self.beta_1, self.beta_2)
        
        # configure input layers
        
        self.image_input = Input(shape = self.img_shape)
        self.omic_input = Input(shape = self.ome_shape)
        
        # build shared latent layer
                                
        self.latent_layer = Lambda(self.Sampling, 
                                   output_shape = (latent_dim,), 
                                   name = 'latent_layer')

        # build image autoencoder
        
        self.I_E = self.ImageEncoder(name = 'image_encoder')
        self.I_D = self.ImageDecoder(name = 'image_decoder')
        self.I_A = self.ImageAutoencoder(name = 'image_vae')
    
        self.I_E.summary()
        self.I_D.summary()
        self.I_A.summary()
        
        self.I_A.compile(optimizer = self.optimizer,
                         loss = self.ImgVAELoss)
        
        # build omic autoencoder
        
        self.O_E = self.OmicEncoder(name = 'omic_encoder')
        self.O_D = self.OmicDecoder(name = 'omic_decoder')
        self.O_A = self.OmicAutoencoder(name = 'omic_vae')
        
        self.O_E.summary()
        self.O_D.summary()
        self.O_A.summary()
        
        self.O_A.compile(optimizer = self.optimizer,
                         loss = self.OmeVAELoss)
        
        # build symbolic transformation tensors
        
        image2omic = self.O_D(self.I_E(self.image_input))
        rec_image = self.I_D(self.O_E(image2omic))

        omic2image = self.I_D(self.O_E(self.omic_input))        
        rec_omic = self.O_D(self.I_E(omic2image))
        
        # build domain transfer models
        
        self.I2O = Model(inputs = self.image_input,
                         outputs = image2omic)
                         
        self.O2I = Model(inputs = self.omic_input,
                         outputs = omic2image)

        # build cycle model
        
        self.C_C = Model(inputs = [self.image_input, self.omic_input],
                         outputs = [rec_image, rec_omic],
                         name = 'cycle_model')
        
        self.C_C.summary()
        
        C_C_loss = ['binary_crossentropy', 
                    'binary_crossentropy']
        
        C_C_weights = [self.lambda_1, 
                       self.lambda_2]
        
        self.C_C.compile(optimizer = self.optimizer,
                         loss = C_C_loss,
                         loss_weights = C_C_weights)
        
        self.Train()
        
        self.EncodeData()
        
        
    def ImageEncoder(self, name = None):
        ''' Encode image into shared latent space '''
        
        x = Conv2D(filters = 16, 
                   kernel_size = 3, 
                   activation = 'relu')(self.image_input)
        
        x = Conv2D(filters = 8, 
                   kernel_size = 3, 
                   activation = 'relu')(x)
        
        x = Conv2D(filters = 4, 
                   kernel_size = 3, 
                   activation = 'relu')(x)
        
        x = Flatten()(x)
        
        x = Dense(self.inter_dim, activation = 'relu')(x)
        
        # reparameterization trick
        
        self.img_z_mean = Dense(self.latent_dim, name = 'img_z_mean')(x)
        self.img_z_log_var = Dense(self.latent_dim, name = 'img_z_log_var')(x)
         
        return Model(inputs = self.image_input, 
                     outputs = self.latent_layer([self.img_z_mean,
                                                  self.img_z_log_var]),
                     name = name)
    
    
    def ImageDecoder(self, name = None):
        ''' Decode latent space into image domain '''

        image_decoder_input = Input(shape = (self.latent_dim,))
        
        x = Dense(np.prod(self.img_shape), 
                  activation = 'relu')(image_decoder_input)
        
        x = Reshape(self.img_shape)(x)
        
        x = Conv2DTranspose(filters = 16,
                            kernel_size = 3,
                            activation='sigmoid',
                            padding='same')(x)
        
        image_output = Conv2DTranspose(filters = 1,
                                       kernel_size = 3,
                                       activation='sigmoid',
                                       padding='same')(x)
        
        return Model(inputs = image_decoder_input, 
                     outputs = image_output, 
                     name = name)
        
    
    def OmicEncoder(self, name = None):
        ''' Encode genomic profile into shared latent space '''
        
        x = Dense(self.inter_dim, activation = 'relu')(self.omic_input)
        
        x = Dense(self.inter_dim, activation = 'relu')(x)
        
        x = Dense(self.latent_dim, activation = 'relu')(x)
        
        # reparameterization trick
        
        self.ome_z_mean = Dense(self.latent_dim, name = 'ome_z_mean')(x)        
        self.ome_z_log_var = Dense(self.latent_dim, name = 'ome_z_log_var')(x)
                
        return Model(inputs = self.omic_input, 
                     outputs = self.latent_layer([self.ome_z_mean,
                                                  self.ome_z_log_var]),
                     name = name)
    
    
    def OmicDecoder(self, name = None):
        ''' Decode latent space into omic domain '''
        
        omic_decoder_input = Input(shape = (self.latent_dim,))
        
        x = Dense(self.inter_dim, activation = 'relu')(omic_decoder_input) 

        x = Dense(self.inter_dim, activation = 'relu')(x) 
        
        omic_output = Dense(self.ome_shape[0], 
                            activation = 'relu')(x)
        
        return Model(inputs = omic_decoder_input, 
                     outputs = omic_output, 
                     name = name)    
    
    
    def ImageAutoencoder(self, name = None):
        ''' Combine image encoder and decoder '''

        return Model(inputs = self.image_input, 
                     outputs = self.I_D(self.I_E(self.image_input)), 
                     name = name)
    
    
    def OmicAutoencoder(self, name = None):
        ''' Combine omic encoder and decoder '''
        
        return Model(inputs = self.omic_input,
                     outputs = self.O_D(self.O_E(self.omic_input)),
                     name = name)
    
          
    def ImgVAELoss(self, y_true, y_pred):
        ''' Compute vae loss for image vae'''
        
        rec_loss = binary_crossentropy(K.flatten(y_true), 
                                       K.flatten(y_pred))
        
        rec_loss *= np.prod(self.img_shape)  
        
        kl_loss = (1 + 
                   self.img_z_log_var - 
                   K.square(self.img_z_mean) - 
                   K.exp(self.img_z_log_var))
        
        kl_loss = K.sum(kl_loss, axis = -1)
        kl_loss *= -0.5
        
        return K.mean(rec_loss + kl_loss)


    def OmeVAELoss(self, y_true, y_pred):
        ''' Compute vae loss for omic vae '''
        
        rec_loss = binary_crossentropy(K.flatten(y_true), 
                                       K.flatten(y_pred))
        
        rec_loss *= np.prod(self.img_shape)         
        
        kl_loss = (1 + 
                   self.ome_z_log_var - 
                   K.square(self.ome_z_mean) - 
                   K.exp(self.ome_z_log_var))
        
        kl_loss = K.sum(kl_loss, axis = -1)
        kl_loss *= -0.5
        
        return K.mean(rec_loss + kl_loss)
        
    
    def Sampling(self, args):
        ''' Reparameterization trick by sampling '''
        
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape = (batch, dim))
        
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    
    def LoadData(self):
        ''' load imaging and omics datasets'''
        
        # TODO: point to actual data
        print('loading data...')
        
    
    def LoadTestData(self):
        ''' Load testing MNIST data sets '''
        
        print('loading data')
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        image_size = x_train.shape[1]
        x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
        x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        
        # subset data for quick prototyping
        
        self.img_train = x_train
        self.ome_train = self.img_train.reshape(-1, np.prod(self.img_train.shape[1:]))
        
        self.img_test = x_test
        self.ome_test = self.img_test.reshape(-1, np.prod(self.img_test.shape[1:]))
        
        self.img_shape = self.img_train.shape[1:]
        self.ome_shape = self.ome_train.shape[1:]
        
        print('training image shape', self.img_train.shape)
        print('training omic shape', self.ome_train.shape)
        
        # save labels for further analysis
        
        pd.DataFrame(y_train).to_csv(os.path.join(self.save_dir, 'labels.csv'),
                                     index = False)


    def InitImageSaver(self):
        ''' create empty array to which to save images '''
                
        self.imgs_to_save_stack = self.img_train[0:self.n_imgs_to_save,:]
        
        self.imgs_to_save = np.concatenate(self.imgs_to_save_stack)
        
    
    def AddReconstructionsToSaver(self):
        ''' add a column of images to a stack '''
        
        print('generating predictions for saving')
        i_a_recon = self.I_A.predict(self.imgs_to_save_stack)
        flat_i_a_recon = np.concatenate(i_a_recon)
        
        i2o = self.I2O.predict(self.imgs_to_save_stack)
        o2i = self.O2I.predict(i2o)
        flat_c_c_recon = np.concatenate(o2i)
        
        img_to_add = np.concatenate((np.ones((flat_i_a_recon.shape[0], 1, 1)),
                                     flat_i_a_recon,
                                     flat_c_c_recon), axis = 1)
        
        self.imgs_to_save = np.concatenate((self.imgs_to_save, img_to_add), 
                                           axis = 1)
    
        save_stack = np.dstack((self.imgs_to_save,
                                self.imgs_to_save,
                                self.imgs_to_save))
        
        save_stack = (save_stack * 255).astype(np.uint8)
        
        save_image = Image.fromarray(save_stack)
        save_image.save(os.path.join(self.save_dir, 
                                     'ImageReconstructions.png'))
         

    def Train(self):
        ''' train XAE model '''
        
        print('training for', self.epochs, 'epochs')
        
        # configure history save file
        
        history_columns = ['epoch',
                           'ImageAutoencoderLoss',
                           'OmicAutoencoderLoss',
                           'CrossChannelLoss']
        
        history_to_save = pd.DataFrame(columns = history_columns)
        
        self.InitImageSaver()
        
        for epoch in range(self.epochs):
                
            print('Epoch {} started'.format(epoch))
                    
            # fit autoencoders
            
            print('fitting image autoencoder')
            I_A_history = self.I_A.fit(x = self.img_train,
                                       y = self.img_train,
                                       epochs = 1,
                                       batch_size = self.batch_size)
            
            print('fitting omic autoencoder')
            O_A_history = self.O_A.fit(x = self.ome_train,
                                       y = self.ome_train,
                                       epochs = 1,
                                       batch_size = self.batch_size)
            
            # fit domain translator
            
            print('fitting domain translator')
            C_C_history = self.C_C.fit(x = [self.img_train, self.ome_train],
                                       y = [self.img_train, self.ome_train],
                                       epochs = 1,
                                       batch_size = self.batch_size)
            # append histories

            history_vals = [epoch,
                            I_A_history.history['loss'][0],
                            O_A_history.history['loss'][0],
                            C_C_history.history['loss'][0]]
            
            history_to_save = history_to_save.append(dict(zip(history_columns, 
                                                              history_vals)),
                                                     ignore_index = True)
            
            self.AddReconstructionsToSaver()

            if self.do_save_model:
                self.SaveModel(epoch)
                                        
                
        history_to_save.to_csv(os.path.join(self.save_dir, 'history.csv'), 
                               index = False)
        
        
    def SaveModel(self, epoch):
        ''' save XAE model '''
        
        print('saving models to file system')
        self.C_C.save_weights(os.path.join(self.save_dir, 
                                           'epoch_' + str(epoch) + '_XAE.h5'))
        
        self.I_A.save_weights(os.path.join(self.save_dir, 
                                           'epoch_' + str(epoch) + 'I_A.h5'))
        
        self.O_A.save_weights(os.path.join(self.save_dir, 
                                           'epoch_' + str(epoch) + 'O_A.h5'))
         

    def EncodeData(self):
        ''' encode XAE model '''
        
        # encode imaging -> latent space
        
        print('encoding images to latent space')
        encoded_images = self.I_E.predict(self.img_train)
        encoded_images = pd.DataFrame(encoded_images)
        encoded_images.to_csv(os.path.join(self.save_dir, 'encodedImages.csv'),
                              index = False)
        
        # encode omics -> latent space
        
        print('encoding omics to latent space')
        encoded_omics = self.O_E.predict(self.ome_train)
        encoded_omics = pd.DataFrame(encoded_omics)
        encoded_omics.to_csv(os.path.join(self.save_dir, 'encodedOmics.csv'),
                             index = False)
        
        # encode image -> omic
        
        print('translating images to omics domain')
        images2omics = self.I2O.predict(self.img_train)
        images2omics = pd.DataFrame(images2omics)
        images2omics.to_csv(os.path.join(self.save_dir, 'images2omics.csv'),
                             index = False)
        
        # encode omic -> image
        
        omics2images = self.O2I.predict(self.ome_train)
        np.save(os.path.join(self.save_dir, 'omics2images.npy'), omics2images)
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'build XAE model')
    parser.add_argument('--learning_rate', type = float, default = 2e-4)
    parser.add_argument('--lambda_1', type = float, default = 10.0)
    parser.add_argument('--lambda_2', type = float, default = 10.0)
    parser.add_argument('--beta_1', type = float, default = 0.5)
    parser.add_argument('--beta_2', type = float, default = 0.99)
    parser.add_argument('--latent_dim', type = int, default = 8)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--epochs', type = int, default = 2)
    parser.add_argument('--n_imgs_to_save', type = int, default = 30)
    parser.add_argument('--save_dir', type = str, default = 'results/test')
    parser.add_argument('--data_dir', type = str, default = 'data/test')
    parser.add_argument('--do_save_model', type = bool, default = False)
    args = parser.parse_args()
    
    xae_model = XAE(learning_rate = args.learning_rate,
                    lambda_1 = args.lambda_1,
                    lambda_2 = args.lambda_2,
                    beta_1 = args.beta_1,
                    beta_2 = args.beta_2,
                    latent_dim = args.latent_dim,
                    batch_size = args.batch_size,
                    epochs = args.epochs,
                    n_imgs_to_save = args.n_imgs_to_save,
                    save_dir = args.save_dir,
                    data_dir = args.data_dir,
                    do_save_model = args.do_save_model)
    