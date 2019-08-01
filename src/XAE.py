'''
XAE: cycle consistent crossed autoencoder for domain translation
'''

import os
import time
import argparse
from random import shuffle

from keras.models import Model
from keras.layers import Input, Dense, Lambda, Reshape, Flatten
from keras.layers import Conv2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.losses import binary_crossentropy, mse
from keras.utils import plot_model
from keras import backend as K
from keras.datasets import mnist

import numpy as np
import pandas as pd
from PIL import Image

os.environ['HDF5_USE_FILE_LOCKING']='FALSE' 


class XAE():
    ''' instantiate a cycle consistent cross-domain autoencoder '''
    
    def __init__(self, 
                 learning_rate = 2e-4,
                 lambda_1 = 10.0,
                 lambda_2 = 10.0,
                 beta_1 = 0.9,
                 beta_2 = 0.99, 
                 latent_dim = 8, 
                 inter_dim = 64,
                 project_dir = '/Users/schau/projects/XAE',
                 data_dir = 'data/test',
                 save_dir = 'results/test',
                 epochs = 2,
                 batch_size = 32,
                 do_save_model = False,
                 do_save_images = True,
                 n_imgs_to_save = 30,
                 is_testing = False,
                 test_rand_add = 0,  # between 0 and 1
                 verbose = 1
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
        
        self.project_dir = project_dir
        self.data_dir = os.path.join(self.project_dir, data_dir)
        self.save_dir = os.path.join(self.project_dir, save_dir)
        
        self.save_dir = save_dir
        self.do_save_model = do_save_model
        self.do_save_images = do_save_images
        
        self.is_testing = is_testing
        self.test_rand_add = test_rand_add
        
        self.verbose = verbose
        
        # create filesystem if not already done
        
        os.makedirs(self.save_dir, exist_ok = True)
        self.date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        
        # load data
        
        if self.is_testing:
            self.LoadTestData()
        else:
            self.LoadData()  # from self.data_dir
        
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
        
        C_C_loss = [self.ImgVAELoss, self.OmeVAELoss]
        
        C_C_weights = [self.lambda_1, self.lambda_2]
        
        self.C_C.compile(optimizer = self.optimizer,
                         loss = C_C_loss,
                         loss_weights = C_C_weights)
        
        self.SavePlotModels()
        
        # train and encode data

        self.Train()
        
        self.EncodeData()


    def SavePlotModels(self):
        ''' save plots of models as pngs '''        
        
        plot_model(self.I_E,
                   to_file = os.path.join(self.save_dir, 'image_encoder.png'),
                   show_shapes = True)
        
        plot_model(self.I_D,
                   to_file = os.path.join(self.save_dir, 'image_decoder.png'),
                   show_shapes = True)
        
        plot_model(self.I_A,
                   to_file = os.path.join(self.save_dir, 'image_AE.png'), 
                   show_shapes = True)
                
        plot_model(self.O_E,
                   to_file = os.path.join(self.save_dir, 'omic_encoder.png'),
                   show_shapes = True)
        
        plot_model(self.O_D,
                   to_file = os.path.join(self.save_dir, 'omic_decoder.png'),
                   show_shapes = True)
        
        plot_model(self.O_A,
                   to_file = os.path.join(self.save_dir, 'omic_AE.png'), 
                   show_shapes = True)        
        
        plot_model(self.I2O,
                   to_file = os.path.join(self.save_dir, 'image2omic.png'),
                   show_shapes = True)
        
        plot_model(self.O2I,
                   to_file = os.path.join(self.save_dir, 'omic2image.png'),
                   show_shapes = True)
        
        plot_model(self.C_C, 
                   to_file = os.path.join(self.save_dir, 'cycle_model.png'), 
                   show_shapes = True)
        
        
        
    def ImageEncoder(self, name = None):
        ''' encode image into shared latent space '''
        
        x = Conv2D(filters = 32, 
                   kernel_size = 3, 
                   activation = 'relu')(self.image_input)
        
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
        
        image_output = Conv2DTranspose(filters = self.img_shape[2],
                                       kernel_size = 3,
                                       activation='sigmoid',
                                       padding='same')(x)
        
        return Model(inputs = image_decoder_input, 
                     outputs = image_output, 
                     name = name)
        
    
    def OmicEncoder(self, name = None):
        ''' encode genomic profile into shared latent space '''
        
        
        x = Dense(self.inter_dim * 32, 
                  activation = 'relu')(self.omic_input)
        
        x = Dense(self.inter_dim * 16, 
                  activation = 'relu')(x)
        
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
        
        x = Dense(self.inter_dim * 32, 
                  activation = 'relu')(x) 
        
        omic_output = Dense(self.ome_shape[0], 
                            activation = 'relu')(x)
        
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
    
    
    def LoadData(self):
        ''' load imaging and omics datasets'''
        
        print('loading data...')
        
        # load and preprocess imaging data
        
        self.img_train = np.load(os.path.join(self.data_dir, 
                                              'images.npy'))
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
        
        print('training image shape', self.img_train.shape)
        print('training omic shape', self.ome_train.shape)
        
    
    def LoadTestData(self):
        ''' load testing MNIST data sets '''
        
        print('loading data')
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        image_size = x_train.shape[1]
        x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
        x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        
        self.img_train = x_train
        self.ome_train = self.img_train.reshape(-1, np.prod(x_train.shape[1:]))
        self.ome_train = self.ome_train
        
        print('original omic train shape', self.ome_train.shape)

        # add "domain specific" information to omic profile
        
        if self.test_rand_add != 0:
            n_samples_to_add = int(self.ome_train.shape[1] * 
                                   self.test_rand_add /
                                   (1 - self.test_rand_add))
            
            shape_to_add = (self.ome_train.shape[0], n_samples_to_add)
            sample_space = np.reshape(self.ome_train, -1)
            
            print('adding samples', n_samples_to_add)
            random_samples = np.random.choice(sample_space, 
                                              np.prod(shape_to_add))
            
            reshape_samples = np.reshape(random_samples, shape_to_add)

            self.ome_train = np.concatenate((self.ome_train,
                                            reshape_samples), axis = 1)

        print('modified omic train shape', self.ome_train.shape)

        self.ome_shape = self.ome_train.shape[1:]
        self.img_shape = self.img_train.shape[1:]

        print('training image shape', self.img_shape)
        print('training omic shape', self.ome_shape)
        
        # save both imaging and omics data
        
        print('saving original data')
        pd.DataFrame(self.ome_train).to_csv(os.path.join(self.save_dir, 
                                                        'input_omics.csv'),
                                            index = False)
        
        np.save(os.path.join(self.save_dir, 'input_images.npy'), 
                self.img_train)
        
        # save labels for further analysis
        
        pd.DataFrame(y_train).to_csv(os.path.join(self.save_dir, 'labels.csv'),
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
                           'CrossChannelLoss']
        
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
            
            self.img_train = self.img_train[img_idx,...]
            self.ome_train = self.ome_train[ome_idx,...]
            
            # fit autoencoders
            
            print('fitting image autoencoder')
            I_A_history = self.I_A.fit(x = self.img_train,
                                       y = self.img_train,
                                       epochs = 1,
                                       batch_size = self.batch_size,
                                       verbose = self.verbose)
            
            print('fitting omic autoencoder')
            O_A_history = self.O_A.fit(x = self.ome_train,
                                       y = self.ome_train,
                                       epochs = 1,
                                       batch_size = self.batch_size,
                                       verbose = self.verbose)
            
            # fit domain translator
            
            print('fitting domain translator')
            C_C_history = self.C_C.fit(x = [self.img_train, self.ome_train],
                                       y = [self.img_train, self.ome_train],
                                       epochs = 1,
                                       batch_size = self.batch_size,
                                       verbose = self.verbose)
            
            # append histories

            history_vals = [epoch,
                            I_A_history.history['loss'][0],
                            O_A_history.history['loss'][0],
                            C_C_history.history['loss'][0]]
            
            history_to_save = history_to_save.append(dict(zip(history_columns, 
                                                              history_vals)),
                                                     ignore_index = True)
            
            history_to_save.to_csv(os.path.join(self.save_dir, 'history.csv'), 
                                   index = False)
            
            if self.do_save_images:
                self.AddReconstructionsToSaver()
                        
        self.SaveModel(epoch)
        
        
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
        
        print('translating omics to imaging domain')
        omics2images = self.O2I.predict(self.ome_train)
        np.save(os.path.join(self.save_dir, 'omics2images.npy'), omics2images)
        
        # reconstructions
        
        print('generating reconstructions')
        recon_images = self.I_A.predict(self.img_train)
        np.save(os.path.join(self.save_dir, 'recon_images.npy'), recon_images)
        
        recon_omics = self.O_A.predict(self.ome_train)
        recon_omics = pd.DataFrame(recon_omics)
        recon_omics.to_csv(os.path.join(self.save_dir, 'recon_omics.csv'),
                           index = False)
        
    
    def WalkFeatureSpace(self):
        ''' walk feature space between domains '''
        
        # TODO: this whole thing
        print('walking feature space...')
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'build XAE model')
    parser.add_argument('--learning_rate', type = float, default = 2e-4)
    parser.add_argument('--lambda_1', type = float, default = 10.0)
    parser.add_argument('--lambda_2', type = float, default = 10.0)
    parser.add_argument('--beta_1', type = float, default = 0.9)
    parser.add_argument('--beta_2', type = float, default = 0.99)
    parser.add_argument('--latent_dim', type = int, default = 8)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--epochs', type = int, default = 5)
    parser.add_argument('--n_imgs_to_save', type = int, default = 30)
    parser.add_argument('--project_dir', type = str, default = '.')
    parser.add_argument('--save_dir', type = str, default = 'results/test')
    parser.add_argument('--data_dir', type = str, default = 'data/test')
    parser.add_argument('--do_save_model', type = bool, default = False)
    parser.add_argument('--do_save_images', type = bool, default = True)
    parser.add_argument('--is_testing', type = bool, default = True)
    parser.add_argument('--test_rand_add', type = float, default = 0)
    parser.add_argument('--verbose', type = int, default = 1)
    
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
                    project_dir = args.project_dir,
                    save_dir = args.save_dir,
                    data_dir = args.data_dir,
                    do_save_model = args.do_save_model,
                    do_save_images = args.do_save_images,
                    is_testing = args.is_testing,
                    test_rand_add = args.test_rand_add,
                    verbose = args.verbose)
    
    
