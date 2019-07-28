'''
XAE: Cycle Consistent Cross-Domain Autoencoder
'''

import os
import time

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Lambda, Reshape, Flatten
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.datasets import mnist

import numpy as np

os.environ['HDF5_USE_FILE_LOCKING']='FALSE' 


class XAE():
    ''' Instantiate a cycle consistent cross-domain autoencoder '''
    
    def __init__(self, 
                 learning_rate = 0.001, 
                 img_shape = (64, 64, 3), 
                 ome_shape = (100,), 
                 latent_dim = 16, 
                 inter_dim = 64,
                 data_dir = 'data/test',
                 save_dir = 'results/test',
                 epochs = 2,
                 batch_size = 1,
                 do_save_model = False
                 ):
        
        # instantiate self parameters
        
        self.lr = learning_rate
        self.img_shape = img_shape
        self.ome_shape = ome_shape
        self.latent_dim = latent_dim
        self.inter_dim = inter_dim
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.data_dir = data_dir
        self.date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        #self.save_dir = os.path.join('results', self.date_time)
        self.save_dir = save_dir
        
        self.do_save_model = do_save_model
        
        os.makedirs(self.save_dir, exist_ok = True)
        
        
        self.LoadData()
        
        # configure input layers
        
        self.image_input = Input(shape = self.img_shape)
        self.omic_input = Input(shape = self.ome_shape)
        
        
        # TODO: configure optimizer alpha, beta decays
        
        self.optimizer = Adam(self.lr)
        
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
        
        
        # build omic autoencoder
        
        self.O_E = self.OmicEncoder(name = 'omic_encoder')
        self.O_D = self.OmicDecoder(name = 'omic_decoder')
        self.O_A = self.OmicAutoencoder(name = 'omic_vae')
        
        self.O_E.summary()
        self.O_D.summary()
        self.O_A.summary()


        # build symbolic transformation tensors
        
        image2omic = self.O_D(self.I_E(self.image_input))
        omic2image = self.I_D(self.O_E(self.omic_input))
        
        rec_image = self.I_D(self.O_E(image2omic))
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
        
        
        # compile models
        
        self.I_A.compile(optimizer = self.optimizer,
                         loss = self.ImgVAELoss)
        
        self.O_A.compile(optimizer = self.optimizer,
                         loss = self.OmeVAELoss)
        
        self.I2O.compile(optimizer = self.optimizer,
                         loss = binary_crossentropy)
        
        self.O2I.compile(optimizer = self.optimizer,
                         loss = binary_crossentropy)
        
        self.C_C.compile(optimizer = self.optimizer,
                         loss = binary_crossentropy)
                
        self.Train()
        
        if self.do_save_model:
            self.SaveModel()
        
        self.Test()
        
        
    
    def ImageEncoder(self, name = None):
        ''' Encode image into shared latent space '''
        
        x = Conv2D(self.latent_dim, 
                   kernel_size = 3, 
                   activation = 'relu')(self.image_input)
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
        print('image decoder shape', self.img_shape)
        image_decoder_input = Input(shape = (self.latent_dim,))
        x = Dense(np.prod(self.img_shape), 
                  activation = 'relu')(image_decoder_input)
        image_output = Reshape(self.img_shape)(x)
        
        return Model(inputs = image_decoder_input, 
                     outputs = image_output, 
                     name = name)
        
    
    def OmicEncoder(self, name = None):
        ''' Encode genomic profile into shared latent space '''
        
        x = Dense(self.latent_dim, activation = 'relu')(self.omic_input)
        x = Dense(self.inter_dim, activation = 'relu')(x)
        
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
        omic_output = Dense(self.ome_shape[0], 
                            activation = 'relu')(omic_decoder_input)
        
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
        ''' Load imaging and omics data sets '''
        
        print('loading data')
        
        # transform imaging and omics data as necessary
        
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        image_size = x_train.shape[1]
        x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
        x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        
        
        self.img_train = x_train[0:300,:]
        self.img_test = x_test[0:100,:]
        
        self.ome_train = x_train.reshape(-1, np.prod(x_train.shape[1:]))[0:300,:]
        self.ome_test = x_test.reshape(-1, np.prod(x_test.shape[1:]))[0:100,:]
        
        self.img_shape = self.img_train.shape[1:]
        self.ome_shape = self.ome_train.shape[1:]
        
        print('training image shape', self.img_train.shape)
        print('training omic shape', self.ome_train.shape)
        print('testing image shape', self.img_test.shape)
        print('testing omic shape', self.ome_test.shape)
        



    def Train(self):
        ''' train XAE model '''
        
        print('training for', self.epochs, 'epochs')
        
        # instantiate arrays for transformed data
        
        #real_A = self.A_test[0]
        #real_B = self.B_test[0]
        #real_A = real_A[np.newaxis, :, :, :]
        #real_B = real_B[np.newaxis, :, :, :]
        
        for epoch in range(self.epochs):
            
            # TODO: shuffle data
            
            print('Epoch {} started'.format(epoch))
            
            # fit autoencoders
            
            print('fitting image autoencoder')
            self.I_A.fit(x = self.img_train,
                         y = self.img_train,
                         epochs = 1,
                         batch_size = self.batch_size,
                         validation_steps = 0)
            
            
            print('fitting omic autoencoder')
            self.O_A.fit(x = self.ome_train,
                         y = self.ome_train,
                         epochs = 1,
                         batch_size = self.batch_size,
                         validation_steps = 0)
                    

            # generate cross-domain predictions
            
            print('generating predictions')
            synthetic_omes = self.I2O.predict(self.img_train)
            synthetic_imgs = self.O2I.predict(self.ome_train)
            
            recreated_imgs = self.O2I.predict(synthetic_omes)
            recreated_omes = self.I2O.predict(synthetic_imgs)
            
            
            # fit domain translators
            
            print('fitting domain translator')
            self.C_C.fit(x = [self.img_train, self.ome_train],
                         y = [recreated_imgs, recreated_omes],
                         epochs = 1,
                         batch_size = self.batch_size,
                         validation_steps = 0)
            
            
            #self.save_tmp_images(real_A, real_B, synthetic_image_A, synthetic_image_B)
            

        #self.saveModel(self.I2O, 200)
        #self.saveModel(self.O2I, 200)


    def SaveModel(self):
        ''' save XAE model '''
        
        print('saving models to file system')
        self.C_C.save_weights(os.path.join(self.save_dir, 'xae.h5'))
        self.I_A.save_weights(os.path.join(self.save_dir, 'image_ae.h5'))
        self.O_A.save_weights(os.path.join(self.save_dir, 'omic_ae.h5'))
         

    def Test(self):
        ''' test XAE model '''
        
        print('testing...')
        


if __name__ == '__main__':

    ccdae_model = XAE()


