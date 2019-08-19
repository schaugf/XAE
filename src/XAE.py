'''
XAE: cycle consistent crossed autoencoder for domain translation
'''

import os
import time
import argparse
from random import shuffle

from keras.models import Model
from keras.layers import Input, Dense, Lambda, Reshape, Flatten
from keras.layers import Conv2D, Conv2DTranspose, Activation, Layer
from keras.optimizers import Adam
from keras.losses import binary_crossentropy, mse
from keras.utils import plot_model
from keras.datasets import mnist
from keras import regularizers
from keras import activations
from keras import backend as K

import numpy as np
import pandas as pd
from PIL import Image

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE' 

# TODO: MNIST, evaluate zeroing noise
# TODO: save reconstructed images as uint
# TODO: training/test split of celeba
# TODO: on-the-fly error correction 
# TODO: implement data augmentation
# TODO: implement cyclic learning rates
# TODO: make sure all data as float32
# TODO: redefine 'image' and 'omic' as A and B
# TODO: add A_name, B_name for easy reference
# TODO: implement minimize encoding separation (L2 of Phi functions)
# TODO: append an alpha to penalize kl contribution
# TODO: balance kl loss as function of size of data (for xent)
# TODO: gate layer for image channels
# TODO: apply trained gate layer to output
# TODO: loss balance, particularly in cycleLoss


class GateLayer(Layer):
    ''' element-wise multiplication gating layer '''
    
    def __init__(self, 
                 output_dim, 
                 kernel_regularizer = None,
                 activation = None,
                 **kwargs):
        
        self.output_dim = output_dim
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activation = activations.get(activation)
        
        super(GateLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name = 'kernel', 
                                      shape = (1, input_shape[1]),
                                      initializer = 'uniform',
                                      trainable = True,
                                      regularizer=self.kernel_regularizer)
        
        super(GateLayer, self).build(input_shape)  
        
    def call(self, x):
        output = x * self.kernel    
        if self.activation is not None:
            output = self.activation(output)
        return output
    

class XAE():
    ''' instantiate a cycle consistent cross-domain autoencoder '''
    
    def __init__(self, 
                 learning_rate = 2e-4,
                 lambda_1 = 10.0,
                 lambda_2 = 10.0,
                 data_class_1 = 'image',
                 data_class_2 = 'omic',
                 beta_1 = 0.9,
                 beta_2 = 0.99, 
                 latent_dim = 8, 
                 inter_dim = 64,
                 project_dir = '/Users/schau/projects/XAE',
                 data_dir = 'data/test',
                 save_dir = 'results/test',
                 epochs = 2,
                 batch_size = 32,
                 do_save_models = 0,
                 do_save_images = 1,
                 do_save_input_data = 0,
                 do_gate_omics = 0,
                 gate_activation = 'tanh',
                 n_imgs_to_save = 30,
                 dataset = 'MNIST',
                 test_rand_add = 0,
                 verbose = 1,
                 omic_activation = 'relu'
                 ):
        
        # instantiate self parameters
        
        self.lr = learning_rate
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.data_class_1 = data_class_1
        self.data_class_2 = data_class_2
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.latent_dim = latent_dim
        self.inter_dim = inter_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_imgs_to_save = n_imgs_to_save
        self.omic_activation = omic_activation
        self.project_dir = project_dir
        self.data_dir = os.path.join(self.project_dir, data_dir)
        self.save_dir = os.path.join(self.project_dir, save_dir)
        
        self.save_dir = save_dir
        self.do_save_models = do_save_models
        self.do_save_images = do_save_images
        self.do_save_input_data = do_save_input_data
        self.do_gate_omics = do_gate_omics
        self.gate_activation = gate_activation
        
        self.dataset = dataset
        self.test_rand_add = test_rand_add
        
        self.verbose = verbose
        
        # create filesystem if not already done
        
        os.makedirs(self.save_dir, exist_ok = True)
        self.date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        
        # load data
                
        if (self.dataset == 'MNIST') | (self.dataset == 'test'):
            self.LoadMNISTData()
        elif self.dataset == 'CelebA':
            self.LoadCelebAData()
        else:
            self.LoadDataFromDir()
            
        if self.test_rand_add != 0:
            self.AddDomainCorruption()
        
        if self.do_save_input_data:
            self.SaveInputData()
            
            
        self.ome_shape = self.ome_train.shape[1:]
        self.img_shape = self.img_train.shape[1:]
        
        print('training image shape', self.img_shape)
        print('training omic shape', self.ome_shape)

        # configure optimizer
        
        self.optimizer = Adam(self.lr, self.beta_1, self.beta_2)
        
        # configure input layers
        
        self.image_input = Input(shape = self.img_shape)
        self.omic_input = Input(shape = self.ome_shape)
        
        # build shared latent layer
                                
        self.latent_layer = Lambda(self.Sampling, 
                                   output_shape = (latent_dim,), 
                                   name = 'latent_layer')
        
        # build gate layer
        
        if self.do_gate_omics:
            gate_r = regularizers.l2(0.01)
            self.gate_layer = GateLayer(self.ome_shape, 
                                        activation = self.gate_activation,
                                        kernel_regularizer = gate_r,
                                        name = 'gate_layer')
        
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

        # build image-to-omic-to-image cycle model
        
        self.I2O2I = Model(inputs = self.image_input,
                           outputs = rec_image,
                           name = 'cycle_model_I2O2I')
        
        self.I2O2I.summary()
        
        
        self.I2O2I.compile(optimizer = self.optimizer,
                           loss = self.ImgVAELoss)        
        
        # build omic-to-image-to-omic
        
        self.O2I2O = Model(inputs = self.omic_input,
                           outputs = rec_omic,
                           name = 'cycle_model_O2I2O')
        
        self.O2I2O.summary()
        
        self.O2I2O.compile(optimizer = self.optimizer,
                           loss = self.OmeVAELoss)
        
        self.SavePlotModels()
        
        # train and encode data

        self.Train()
        
        print('saving gating layer')
        
        gate_weights = self.O_E.layers[1].get_weights()[0][0]
        
        np.savetxt(os.path.join(self.save_dir, 
                                'gate_weights.csv'), 
                   gate_weights, 
                   delimiter = ',')

        self.SaveEncodedData()
        self.SaveReconstructedData()
        self.SaveTranslatedData()
        
        

    def SavePlotModels(self):
        ''' save plots of models as pngs '''   
        
        model_dir = os.path.join(self.save_dir, 'model_plots')
        os.makedirs(model_dir, exist_ok = True)
        
        plot_model(self.I_E, show_shapes = True,
                   to_file = os.path.join(model_dir, 'image_encoder.png'))
        
        plot_model(self.I_D, show_shapes = True,
                   to_file = os.path.join(model_dir, 'image_decoder.png'))
        
        plot_model(self.I_A, show_shapes = True,
                   to_file = os.path.join(model_dir, 'image_AE.png'))
                
        plot_model(self.O_E, show_shapes = True,
                   to_file = os.path.join(model_dir, 'omic_encoder.png'))
        
        plot_model(self.O_D, show_shapes = True,
                   to_file = os.path.join(model_dir, 'omic_decoder.png'))
        
        plot_model(self.O_A, show_shapes = True,
                   to_file = os.path.join(model_dir, 'omic_AE.png'))        
        
        plot_model(self.I2O, show_shapes = True,
                   to_file = os.path.join(model_dir, 'image2omic.png'))
        
        plot_model(self.O2I, show_shapes = True,
                   to_file = os.path.join(model_dir, 'omic2image.png'))
        
        plot_model(self.I2O2I, show_shapes = True,
                   to_file = os.path.join(model_dir, 'cycle_modeO_I2O2I.png'))
        
        plot_model(self.O2I2O, show_shapes = True,
                   to_file = os.path.join(model_dir, 'cycle_modeO_O2I2O.png'))
        
        
    def ImageEncoder(self, name = None):
        ''' encode image into shared latent space '''
        
        x = Conv2D(filters = 64, 
                   kernel_size = 3, 
                   activation = 'relu')(self.image_input)
        
        x = Conv2D(filters = 32, 
                   kernel_size = 3, 
                   activation = 'relu')(x)
        
        x = Conv2D(filters = 16, 
                   kernel_size = 3, 
                   activation = 'relu')(x)
        
        x = Conv2D(filters = 8, 
                   kernel_size = 3, 
                   activation = 'relu')(x)
        
        x = Conv2D(filters = 4, 
                   kernel_size = 3, 
                   activation = 'relu')(x)
        
        x = Flatten()(x)
        
        x = Dense(self.inter_dim, 
                  activation = 'relu')(x)
        
        # reparameterization trick
        
        self.img_z_mean = Dense(self.latent_dim, name = 'img_z_mean')(x)
        self.img_z_log_var = Dense(self.latent_dim, name = 'img_z_log_var')(x)
         
        return Model(inputs = self.image_input, 
                     outputs = self.latent_layer([self.img_z_mean,
                                                  self.img_z_log_var]),
                     name = name)
    
    
    def ImageDecoder(self, name = None):
        ''' decode latent space into image domain '''

        image_decoder_input = Input(shape = (self.latent_dim,))
        
        x = Dense(np.prod(self.img_shape), 
                  activation = 'relu')(image_decoder_input)
        
        x = Reshape(self.img_shape)(x)
        
        x = Conv2DTranspose(filters = 4,
                            kernel_size = 3,
                            activation = 'relu',
                            padding = 'same')(x)
        
        x = Conv2DTranspose(filters = 8,
                            kernel_size = 3,
                            activation = 'relu',
                            padding = 'same')(x)
        
        x = Conv2DTranspose(filters = 16,
                            kernel_size = 3,
                            activation = 'relu',
                            padding = 'same')(x)
        
        x = Conv2DTranspose(filters = 32,
                            kernel_size = 3,
                            activation = 'relu',
                            padding = 'same')(x)
        
        x = Conv2DTranspose(filters = 64,
                            kernel_size = 3,
                            activation = 'relu',
                            padding = 'same')(x)
        
        image_output = Conv2DTranspose(filters = self.img_shape[2],
                                       kernel_size = 3,
                                       activation='sigmoid',
                                       padding='same')(x)
        
        return Model(inputs = image_decoder_input, 
                     outputs = image_output, 
                     name = name)
        
    
    def OmicEncoder(self, name = None):
        ''' encode genomic profile into shared latent space '''
        
        if self.do_gate_omics:
            print('gating omics input with input shape', self.ome_shape[0])
            x = self.gate_layer(self.omic_input)
            x = Dense(self.inter_dim * 16, 
                      activation = 'relu')(x)
        else:
            x = Dense(self.inter_dim * 16, 
                      activation = 'relu')(self.omic_input)
            
        x = Dense(self.inter_dim * 8, 
                  activation = 'relu')(x)
        
        x = Dense(self.inter_dim * 4, 
                  activation = 'relu')(x)
        
        x = Dense(self.inter_dim * 2, 
                  activation = 'relu')(x)
        
        x = Dense(self.latent_dim, 
                  activation = 'relu')(x)
        
        # reparameterization trick
        
        self.ome_z_mean = Dense(self.latent_dim, name = 'ome_z_mean')(x)        
        self.ome_z_log_var = Dense(self.latent_dim, name = 'ome_z_log_var')(x)
                
        return Model(inputs = self.omic_input, 
                     outputs = self.latent_layer([self.ome_z_mean,
                                                  self.ome_z_log_var]),
                     name = name)
    
    
    def OmicDecoder(self, name = None):
        ''' decode latent space into omic domain '''
        
        omic_decoder_input = Input(shape = (self.latent_dim,))
        
        x = Dense(self.inter_dim * 2, 
                  activation = 'relu')(omic_decoder_input) 

        x = Dense(self.inter_dim * 4, 
                  activation = 'relu')(x) 
        
        x = Dense(self.inter_dim * 8, 
                  activation = 'relu')(x) 
        
        x = Dense(self.inter_dim * 16, 
                  activation = 'relu')(x) 
        
        if self.do_gate_omics:
            x = Dense(self.ome_shape[0], 
                      activation = 'relu')(x)
            omic_output = self.gate_layer(x)
            
        else:
            omic_output = Dense(self.ome_shape[0], 
                                activation = self.omic_activation)(x)
        
        return Model(inputs = omic_decoder_input, 
                     outputs = omic_output, 
                     name = name)    
    
    
    def ImageAutoencoder(self, name = None):
        ''' combine image encoder and decoder '''

        return Model(inputs = self.image_input, 
                     outputs = self.I_D(self.I_E(self.image_input)), 
                     name = name)
    
    
    def OmicAutoencoder(self, name = None):
        ''' combine omic encoder and decoder '''
        
        return Model(inputs = self.omic_input,
                     outputs = self.O_D(self.O_E(self.omic_input)),
                     name = name)
    
          
    def ImgCycleLoss(self, y_true, y_pred):
        ''' loss for cyclic transformation '''
        
        rec_loss = binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
        rec_loss *= np.prod(self.img_shape)
        
        kl_loss = (1 + self.img_z_log_var - 
                   K.square(self.img_z_mean) - K.exp(self.img_z_log_var))
                   
        kl_loss = K.sum(kl_loss, axis = -1)
        kl_loss *= -0.5   
        
        img_encoding = self.latent_layer([self.img_z_mean, 
                                          self.img_z_log_var])  
      
        ome_encoding = self.latent_layer([self.ome_z_mean, 
                                          self.ome_z_log_var])
        
        mutual_encoding_loss = mse(img_encoding, ome_encoding)
        
        return K.mean(rec_loss + kl_loss + mutual_encoding_loss)
    
    
    def OmeCycleLoss(self, y_true, y_pred):
        ''' loss for cyclic transformation '''
        
        rec_loss = binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
        rec_loss *= np.prod(self.ome_shape)
        
        kl_loss = (1 + self.ome_z_log_var - 
                   K.square(self.ome_z_mean) - K.exp(self.ome_z_log_var))
                   
        kl_loss = K.sum(kl_loss, axis = -1)
        kl_loss *= -0.5     
        
        img_encoding = self.latent_layer([self.img_z_mean, 
                                          self.img_z_log_var])  
      
        ome_encoding = self.latent_layer([self.ome_z_mean, 
                                          self.ome_z_log_var])
        
        mutual_encoding_loss = mse(img_encoding, ome_encoding)
        
        return K.mean(rec_loss + kl_loss + mutual_encoding_loss)
    
    
    def ImgVAELoss(self, y_true, y_pred):
        ''' compute vae loss for image vae'''
        
        rec_loss = binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
        rec_loss *= np.prod(self.img_shape)
        
        kl_loss = (1 + self.img_z_log_var - 
                   K.square(self.img_z_mean) - K.exp(self.img_z_log_var))
                   
        kl_loss = K.sum(kl_loss, axis = -1)
        kl_loss *= -0.5       
        
        return K.mean(rec_loss + kl_loss)


    def OmeVAELoss(self, y_true, y_pred):
        ''' compute vae loss for omic vae '''
                
        rec_loss = mse(K.flatten(y_true), K.flatten(y_pred))
        rec_loss *= np.prod(self.ome_shape)
        
        kl_loss = (1 + self.ome_z_log_var - 
                   K.square(self.ome_z_mean) - K.exp(self.ome_z_log_var))
                   
        kl_loss = K.sum(kl_loss, axis = -1)
        kl_loss *= -0.5     
                                            
        return K.mean(rec_loss + kl_loss)
        
    
    def Sampling(self, args):
        ''' reparameterization trick by sampling '''
        
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape = (batch, dim))
        
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    
    def LoadDataFromDir(self):
        ''' load imaging and omics datasets'''
        
        print('loading data from', self.data_dir)
                
        self.img_train = np.load(os.path.join(self.data_dir, 
                                              'images.npy'))
        
        print('loaded images with shape', self.img_train.shape)
        
        if self.img_train.dtype == np.uint16:
            self.img_train = self.img_train.astype(np.float32) / (2**16 - 1)
        elif self.img_train.dtype == np.uint8:
            self.img_train = self.img_train.astype(np.float32) / (2**8 - 1)
        
        # load and preprocess omics data
        
        self.ome_train = pd.read_csv(os.path.join(self.data_dir, 'omics.csv'))
        
        self.ome_train = self.ome_train.drop(self.ome_train.columns[0:5], 
                                             axis = 1)
        
        self.ome_train = np.array(self.ome_train).astype(np.float32)
        self.ome_train = np.log(self.ome_train + 1)
        self.ome_train = self.ome_train - self.ome_train.min()
        self.ome_train = self.ome_train / self.ome_train.max()
        
        self.img_shape = self.img_train.shape[1:]
        self.ome_shape = self.ome_train.shape[1:]
        
    
    def LoadCelebAData(self):
        ''' load testing celebA dataset '''

        print('loading CelebA dataset...')
        
        self.omic_activation = 'tanh'
        
        # load raw data
        self.img_train = np.load('data/celeb/celebA.npy')
        self.img_train = self.img_train.astype('float32') / 255
        
        self.ome_train = np.load('data/celeb/img_annot.npy')
        
        # split into training/test
        
        # set shape params
        self.img_shape = self.img_train.shape[1:]
        self.ome_shape = self.ome_train.shape[1:]

 
    def AddDomainCorruption(self):
        ''' append domain-specific corruption '''
        
        if self.test_rand_add != 0:
            print('corrupting omics domain')
            print('original omic train shape', self.ome_train.shape)

            n_samples_to_add = int(self.ome_train.shape[1] * 
                                   self.test_rand_add /
                                   (1 - self.test_rand_add))
            
            train_shape_to_add = (self.ome_train.shape[0], n_samples_to_add)
            test_shape_to_add = (self.ome_test.shape[0], n_samples_to_add)
            
            sample_space = np.reshape(self.ome_train, -1)

            train_samples = np.random.choice(sample_space, 
                                             np.prod(train_shape_to_add))
            
            test_samples = np.random.choice(sample_space, 
                                             np.prod(test_shape_to_add))
            
            train_samples = np.reshape(train_samples, train_shape_to_add)
            test_samples = np.reshape(test_samples, test_shape_to_add)
        
            self.ome_train = np.concatenate((self.ome_train, train_samples), 
                                            axis = 1)
            
            self.ome_test = np.concatenate((self.ome_test, test_samples), 
                                            axis = 1)
            
            print('corrupted omic train shape', self.ome_train.shape)
            print('corrupted omic test shape', self.ome_test.shape)
            
            

    def SaveInputData(self):
        ''' save the original data'''
    
        print('saving original data')
        pd.DataFrame(self.ome_train).to_csv(os.path.join(self.save_dir, 
                                                        'input_omics.csv'),
                                            index = False)
        
        np.save(os.path.join(self.save_dir, 'input_images.npy'), 
                self.img_train)

        
    
    def LoadMNISTData(self):
        ''' load testing MNIST data sets '''
        
        print('loading MNIST data')
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        image_size = x_train.shape[1]
        x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
        x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        
        self.img_train = x_train
        self.img_test = x_test
        self.ome_train = self.img_train.reshape(-1, np.prod(x_train.shape[1:]))
        self.ome_test = self.img_test.reshape(-1, np.prod(x_test.shape[1:]))

        if self.dataset == 'test':
            self.img_train = self.img_train[0:200]
            self.img_test = self.img_test[0:200]
            self.ome_train = self.ome_train[0:200]
            self.ome_test = self.ome_test[0:200]
        
        # save labels
        pd.DataFrame(y_train).to_csv(os.path.join(self.save_dir, 
                                                  'labels_train.csv'),
                                     index = False)
            
        pd.DataFrame(y_test).to_csv(os.path.join(self.save_dir,
                                                 'labels_test.csv'),
                                    index = False)


    def InitImageSaver(self):
        ''' create empty array to which to save images '''
                
        self.imgs_to_save_stack = self.img_train[0:self.n_imgs_to_save,:]
        
        self.imgs_to_save = np.concatenate(self.imgs_to_save_stack)
        
        print('images to save have shape', self.imgs_to_save.shape)
        
    
    def AddReconstructionsToSaver(self):
        ''' add a column of images to a stack '''
        
        # TODO: this for omics domain
        print('generating predictions for saving')
        i_a_recon = self.I_A.predict(self.imgs_to_save_stack)
        flat_i_a_recon = np.concatenate(i_a_recon)
        
        i2o = self.I2O.predict(self.imgs_to_save_stack)
        o2i = self.O2I.predict(i2o)
        
        flat_c_c_recon = np.concatenate(o2i)
        
        img_to_add = np.concatenate((np.ones((flat_i_a_recon.shape[0], 
                                              1, 
                                              flat_i_a_recon.shape[2])),
                                     flat_i_a_recon,
                                     flat_c_c_recon), axis = 1)
        
        self.imgs_to_save = np.concatenate((self.imgs_to_save, img_to_add), 
                                           axis = 1)
        
        self.SaveReconstructionImage()
        
        
    def SaveReconstructionImage(self):
        ''' save panel of reconstructed images '''
        
        image_to_save = (self.imgs_to_save * 255).astype(np.uint8)
        
        if image_to_save.shape[2] == 3:  # color image
            save_file_name = 'image_reconstruction.jpg'
            save_image = Image.fromarray(image_to_save)
            save_image.save(os.path.join(self.save_dir, save_file_name))
        else:
            for i in range(image_to_save.shape[2]):
                save_file_name = 'channel_' + str(i) + '_reconstruction.jpg'
                single_channel = image_to_save[...,i]
                save_image = Image.fromarray(single_channel)
                save_image.save(os.path.join(self.save_dir, save_file_name))
            
         

    def Train(self):
        ''' train XAE model '''
        
        print('training for', self.epochs, 'epochs')
        
        # configure history save file
        
        history_columns = ['epoch',
                           'ImageAutoencoderLoss',
                           'OmicAutoencoderLoss',
                           'I2O2ILoss',
                           'O2I2OLoss',
                           'ImageAutoencoderLossEval',
                           'OmicAutoencoderLossEval',
                           'I2O2ILossEval',
                           'O2I2OLossEval']
        
        history_to_save = pd.DataFrame(columns = history_columns)
        
        # indexes to shuffle
        
        img_idx = [i for i in range(len(self.img_train))]
        ome_idx = [i for i in range(len(self.ome_train))]
            
        self.InitImageSaver()
        
        for epoch in range(self.epochs):
                
            print('Epoch {} started'.format(epoch))
            
            # shuffle indices
            
            shuffle(img_idx)
            shuffle(ome_idx)
            
            # fit autoencoders

            print('fitting image autoencoder')
            I_A_history = self.I_A.fit(x = self.img_train[img_idx,...],
                                       y = self.img_train[img_idx,...],
                                       epochs = 1,
                                       batch_size = self.batch_size,
                                       verbose = self.verbose)

            
            print('fitting omic autoencoder')
            O_A_history = self.O_A.fit(x = self.ome_train[ome_idx,...],
                                       y = self.ome_train[ome_idx,...],
                                       epochs = 1,
                                       batch_size = self.batch_size,
                                       verbose = self.verbose)
            
            # fit domain translators
            
            print('fitting domain translator I2O2I')
            I2O2I_history = self.I2O2I.fit(x = self.img_train[img_idx,...], 
                                           y = self.img_train[img_idx,...], 
                                           epochs = 1,
                                           batch_size = self.batch_size,
                                           verbose = self.verbose)
            
            print('fitting domain translator O2I2O')
            O2I2O_history = self.O2I2O.fit(x = self.ome_train[ome_idx,...],
                                           y = self.ome_train[ome_idx,...],
                                           epochs = 1,
                                           batch_size = self.batch_size,
                                           verbose = self.verbose)            
            
            # evaluate models on testing data
            
            print('evaluating trained models on test set')
            I_A_eval = self.I_A.evaluate(x = self.img_test,
                                         y = self.img_test)
            
            O_A_eval = self.O_A.evaluate(x = self.ome_test,
                                         y = self.ome_test)
            
            I2O2I_eval = self.I2O2I.evaluate(x = self.img_test,
                                             y = self.img_test)
                        
            O2I2O_eval = self.O2I2O.evaluate(x = self.ome_test,
                                             y = self.ome_test)
            
            # append histories

            history_vals = [epoch,
                            I_A_history.history['loss'][0],
                            O_A_history.history['loss'][0],
                            I2O2I_history.history['loss'][0],
                            O2I2O_history.history['loss'][0],
                            I_A_eval,
                            O_A_eval,
                            I2O2I_eval,
                            O2I2O_eval]
                        
            history_to_save = history_to_save.append(dict(zip(history_columns, 
                                                              history_vals)),
                                                     ignore_index = True)
            
            history_to_save.to_csv(os.path.join(self.save_dir, 'history.csv'), 
                                   index = False)
            
            if self.do_save_images:
                self.AddReconstructionsToSaver()
        
        if self.do_save_models:
            self.SaveModels(epoch)
            
        if self.do_gate_omics:
            print('saving omics gate')
        
        
    def SaveModels(self, epoch):
        ''' save XAE model '''
        
        print('saving models to file system')
        
        model_dir = os.path.join(self.save_dir, 'model_plots')
        os.makedirs(model_dir, exist_ok = True)
        
        self.I2O2I.save_weights(os.path.join(model_dir, 
                                             'epoch_' + str(epoch) + 
                                             '_I2O2I.h5'))
        
        self.O2I2O.save_weights(os.path.join(model_dir, 
                                             'epoch_' + str(epoch) + 
                                             '_O2I2O.h5'))
        
        self.I_A.save_weights(os.path.join(model_dir, 
                                           'epoch_' + str(epoch) + '_I_A.h5'))
        
        self.O_A.save_weights(os.path.join(model_dir, 
                                           'epoch_' + str(epoch) + '_O_A.h5'))
         

    def SaveEncodedData(self):
        ''' encode XAE model '''
                
        encoded_dir = os.path.join(self.save_dir, 'encodings')
        os.makedirs(encoded_dir, exist_ok = True)
        
        # encode imaging -> latent space
        
        print('encoding images to latent space')
        encoded_images_train = self.I_E.predict(self.img_train)
        encoded_images_train = pd.DataFrame(encoded_images_train)
        encoded_images_train.to_csv(os.path.join(encoded_dir, 
                                           'encodedImages_train.csv'),
                                    index = False)
        
        encoded_images_test = self.I_E.predict(self.img_test)
        encoded_images_test = pd.DataFrame(encoded_images_test)
        encoded_images_test.to_csv(os.path.join(encoded_dir, 
                                           'encodedImages_test.csv'),
                                   index = False)
            
        # encode omics -> latent space
        
        print('encoding omics to latent space')
        encoded_omics_train = self.O_E.predict(self.ome_train)
        encoded_omics_train = pd.DataFrame(encoded_omics_train)
        encoded_omics_train.to_csv(os.path.join(encoded_dir, 
                                          'encodedOmics_train.csv'),
                                   index = False)
            
        encoded_omics_test = self.O_E.predict(self.ome_test)
        encoded_omics_test = pd.DataFrame(encoded_omics_test)
        encoded_omics_test.to_csv(os.path.join(encoded_dir, 
                                          'encodedOmics_test.csv'),
                                  index = False)
        
        
    def SaveReconstructedData(self):
        ''' save autoencoder and cycled reconstructions '''
                    
        reconstructed_dir = os.path.join(self.save_dir, 'reconstructions')
        os.makedirs(reconstructed_dir, exist_ok = True)
        
        # autoencoder reconstructions
        
        print('generating image reconstructions')
        recon_images_train = self.I_A.predict(self.img_train)
        np.save(os.path.join(reconstructed_dir, 'recon_images_train.npy'), 
                             recon_images_train)
        
        recon_images_test = self.I_A.predict(self.img_test)
        np.save(os.path.join(reconstructed_dir, 'recon_images_test.npy'), 
                             recon_images_test)
        
        print('generating omic reconstructions')
        recon_omics_train = self.O_A.predict(self.ome_train)
        recon_omics_train = pd.DataFrame(recon_omics_train)
        recon_omics_train.to_csv(os.path.join(reconstructed_dir, 
                                              'recon_omics_train.csv'),
                                 index = False)
        
        recon_omics_test = self.O_A.predict(self.ome_test)
        recon_omics_test = pd.DataFrame(recon_omics_test)
        recon_omics_test.to_csv(os.path.join(reconstructed_dir, 
                                              'recon_omics_test.csv'),
                                 index = False)


        # save cycled reconstructions
        
        print('generating image cycled reconstructions')
        cycle_images_train = self.I2O2I.predict(self.img_train)
        np.save(os.path.join(reconstructed_dir, 'cycled_images_train.npy'), 
                             cycle_images_train)
        
        cycle_images_test = self.I2O2I.predict(self.img_test)
        np.save(os.path.join(reconstructed_dir, 'cycled_images_test.npy'), 
                             cycle_images_test)
        
        print('generating omic cycled reconstructions')
        cycle_omics_train = self.O2I2O.predict(self.ome_train)
        cycle_omics_train = pd.DataFrame(cycle_omics_train)
        cycle_omics_train.to_csv(os.path.join(reconstructed_dir, 
                                              'cycle_omics_train.csv'),
                                 index = False)
        
        cycle_omics_test = self.O2I2O.predict(self.ome_test)
        cycle_omics_test = pd.DataFrame(cycle_omics_test)
        cycle_omics_test.to_csv(os.path.join(reconstructed_dir, 
                                              'cycle_omics_test.csv'),
                                 index = False)
        
    
    def SaveTranslatedData(self):
        ''' translate imaging and omics into corresponding target domains '''
        
        translated_dir = os.path.join(self.save_dir, 'translations')
        os.makedirs(translated_dir, exist_ok = True)
        
        # domain translation image -> omic
        
        print('translating images to omics domain')
        images2omics_train = self.I2O.predict(self.img_train)
        images2omics_train = pd.DataFrame(images2omics_train)
        images2omics_train.to_csv(os.path.join(translated_dir, 
                                         'images2omics_train.csv'),
                             index = False)
            
        images2omics_test = self.I2O.predict(self.img_test)
        images2omics_test = pd.DataFrame(images2omics_test)
        images2omics_test.to_csv(os.path.join(translated_dir, 
                                              'images2omics_test.csv'),
                             index = False)            
        
        # encode omic -> image
        
        print('translating omics to imaging domain')
        omics2images_train = self.O2I.predict(self.ome_train)
        np.save(os.path.join(translated_dir, 'omics2images_train.npy'), 
                omics2images_train)
        
        omics2images_test = self.O2I.predict(self.ome_test)
        np.save(os.path.join(translated_dir, 'omics2images_test.npy'), 
                omics2images_test)
        
        
    def WalkFeatureSpace(self):
        ''' walk feature space between domains '''
        # TODO: this whole thing
        # for each latent variable:
        print('walking feature space...')
        
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'build XAE model')
    
    parser.add_argument('--learning_rate', type = float, default = 2e-4)
    parser.add_argument('--lambda_1', type = float, default = 10.0)
    parser.add_argument('--lambda_2', type = float, default = 10.0)
    parser.add_argument('--data_class_1', type = str, default = 'image')
    parser.add_argument('--data_class_2', type = str, default = 'omic')
    parser.add_argument('--beta_1', type = float, default = 0.9)
    parser.add_argument('--beta_2', type = float, default = 0.99)
    parser.add_argument('--latent_dim', type = int, default = 8)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--epochs', type = int, default = 1)
    parser.add_argument('--n_imgs_to_save', type = int, default = 30)
    parser.add_argument('--project_dir', type = str, default = '.')
    parser.add_argument('--save_dir', type = str, default = 'results/test')
    parser.add_argument('--data_dir', type = str, default = 'data/test')
    parser.add_argument('--do_save_models', type = int, default = 0)
    parser.add_argument('--do_save_images', type = int, default = 1)
    parser.add_argument('--do_save_input_data', type = int, default = 0)
    parser.add_argument('--do_gate_omics', type = int, default = 1)
    parser.add_argument('--gate_activation', type = str, default = 'sigmoid')
    parser.add_argument('--dataset', type = str, default = 'test')
    parser.add_argument('--test_rand_add', type = float, default = 0.2)
    parser.add_argument('--verbose', type = int, default = 1)    
    parser.add_argument('--omic_activation', type = str, default = 'relu')
    
    args = parser.parse_args()
    
        
    xae_model = XAE(learning_rate = args.learning_rate,
                    lambda_1 = args.lambda_1,
                    lambda_2 = args.lambda_2,
                    data_class_1 = args.data_class_1,
                    data_class_2 = args.data_class_2,
                    beta_1 = args.beta_1,
                    beta_2 = args.beta_2,
                    latent_dim = args.latent_dim,
                    batch_size = args.batch_size,
                    epochs = args.epochs,
                    n_imgs_to_save = args.n_imgs_to_save,
                    project_dir = args.project_dir,
                    save_dir = args.save_dir,
                    data_dir = args.data_dir,
                    do_save_models = args.do_save_models,
                    do_save_images = args.do_save_images,
                    do_save_input_data = args.do_save_input_data,
                    do_gate_omics = args.do_gate_omics,
                    gate_activation = args.gate_activation,
                    dataset = args.dataset,
                    test_rand_add = args.test_rand_add,
                    verbose = args.verbose,
                    omic_activation = args.omic_activation)
    
    