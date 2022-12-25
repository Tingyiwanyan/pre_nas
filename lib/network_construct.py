import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tensorflow as tf
import cv2
import skimage.io
from skimage.transform import resize
from imgaug import augmenters as iaa
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer,LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Dropout, \
    Activation, ZeroPadding2D, BatchNormalization, Flatten, \
    Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,GlobalAveragePooling2D,Concatenate, ReLU,\
    LeakyReLU,Reshape, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, \
    EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import regularizers
from tqdm import tqdm
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
import keras.backend as K
from tensorflow.keras.metrics import categorical_accuracy,top_k_categorical_accuracy
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


class network_construction():
    def __init__(self):
        self.batch_size = 128
        self.seed = 42
        self.input_shape = (64, 64, 3)  ##desired shape of the image for resizing purposes
        self.val_sample = 0.1
        self.resnet_arch()
        #self.strategy = tf.distribute.MirroredStrategy()

    def resnet_arch(self):
        """
        Resnet50 architecture
        """
        #self.stages = ["stage_2","stage_3","stage_4","stage_5"]

        self.stage_2 = {}
        self.stage_2["structure_list"] = "conv identity identity"
        self.stage_2["conv"] = "64 64 256"
        self.stage_2["identity"] = "64 64 256"
        self.stage_2["identity"] = "64 64 256"

        self.stage_3 = {}
        self.stage_3["structure_list"] = "conv identity identity identity"
        self.stage_3["conv"] = "128 128 512"
        self.stage_3["identity"] = "128 128 512"
        self.stage_3["identity"] = "128 128 512"
        self.stage_3["identity"] = "128 128 512"

        self.stage_4 = {}
        self.stage_4["structure_list"] = "conv identity identity identity identity identity"
        self.stage_4["conv"] = "256 256 1024"
        self.stage_4["identity"] = "256 256 1024"
        self.stage_4["identity"] = "256 256 1024"
        self.stage_4["identity"] = "256 256 1024"
        self.stage_4["identity"] = "256 256 1024"
        self.stage_4["identity"] = "256 256 1024"

        self.stage_5 = {}
        self.stage_5["structure_list"] = "conv identity identity"
        self.stage_5["conv"] = "512 512 2048"
        self.stage_5["identity"] = "512 512 2048"
        self.stage_5["identity"] = "512 512 2048"





    def identity_block(self, input_x, f, filters, stage, block):
        """
        Implementation of the identity block

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network

        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        conv_name_base = 'identity_block' + '_' + str(stage) + '_' + block
        bn_name_base = 'bn' + '_' + str(stage) + '_' + block
        activation_name_base = 'activation_identity' + '_' + str(stage) + '_' + block
        addition_base = 'addition_identity' + '_' + str(stage) + '_' + block +'_shortcut'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value. You'll need this later to add back to the main path.
        #X_init = Input(input_x.shape[1:])
        X_shortcut = input_x
        #X_shortcut = input_x

        # First component of main path
        X = Conv2D(filters=F1, kernel_regularizer=regularizers.L2(1e-4), kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '_a',
                   kernel_initializer=glorot_uniform(seed=0))(input_x)
        X = BatchNormalization(axis=3, name=bn_name_base + '_a')(X)
        X = Activation('relu', name=activation_name_base+ '_a')(X)

        # Second component of main path
        X = Conv2D(filters=F2, kernel_regularizer=regularizers.L2(1e-4), kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '_b',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '_b')(X)
        X = Activation('relu', name=activation_name_base+ '_b')(X)

        # Third component of main path
        X = Conv2D(filters=F3, kernel_regularizer=regularizers.L2(1e-4), kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '_c',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '_c')(X)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X = Add(name=addition_base)([X, X_shortcut])
        X = Activation('relu', name=activation_name_base+ '_shortcut')(X)

        return X
        #return  Model(inputs=X_init, outputs=X, name="Identity_block"+str(stage)+num)

    def convolutional_block(self, input_x, f, filters, stage, block, s=2):
        """
        Implementation of the convolutional block

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        s -- Integer, specifying the stride to be used

        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        #conv_name_base = 'conv_' + str(stage) + block
        #bn_name_base = 'bn_' + str(stage) + block

        conv_name_base = 'conv_block_' + str(stage) + '_' + block
        bn_name_base = 'bn_' + str(stage) + '_' + block
        activation_name_base = 'activation_conv_' + str(stage) + '_' + block
        addition_base = 'addition_conv_' + str(stage) + '_' + block

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value
        #X_init = Input(input_x.shape[1:])
        X_shortcut = input_x

        ##### MAIN PATH #####
        # First component of main path
        X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base+'_a',kernel_regularizer=regularizers.L2(1e-4),
                   kernel_initializer=glorot_uniform(seed=0))(input_x)
        X = BatchNormalization(axis=3, name=bn_name_base + '_a')(X)
        X = Activation('relu', name=activation_name_base + '_a')(X)

        # Second component of main path
        X = Conv2D(F2, (f, f), strides=(1, 1), padding='same', kernel_regularizer=regularizers.L2(1e-4),
                   name=conv_name_base + '_b',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '_b')(X)
        X = Activation('relu', name=activation_name_base + '_b')(X)

        # Third component of main path
        X = Conv2D(F3, (1, 1), strides=(1, 1), padding='valid', kernel_regularizer=regularizers.L2(1e-4),
                   name=conv_name_base + '_c',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '_c')(X)

        ##### SHORTCUT PATH ####
        X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), padding='valid', kernel_regularizer=regularizers.L2(1e-4),
                            name=conv_name_base + '_shortcut',
                            kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '_shortcut')(X_shortcut)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X = Add(name=addition_base)([X, X_shortcut])
        X = Activation('relu', name=activation_name_base + '_shortcut')(X)

        return X
        #return Model(inputs=X_init, outputs=X, name="conv_block"+str(stage)+num)


    def update_layer(self, X, name, filters_, stage_,nums, s_):
        if name == "conv":
            return self.convolutional_block(X, f=3, filters=filters_, stage=stage_, block=str(nums),s=s_)
        if name == "identity":
            return self.identity_block(X, 3, filters=filters_, stage=stage_, block=str(nums))



    def stack_architecture(self, structure_list, input_shape=(64, 64, 3)):
        #with self.strategy.scope():
        X_input = Input(input_shape)

        X = ZeroPadding2D((3, 3))(X_input)

        X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_regularizer=regularizers.L2(1e-4),
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        for stage_name in structure_list:
            if stage_name == "stage_2":
                stage_list = self.stage_2
                s_=1
                s_stage=2
            if stage_name == "stage_3":
                stage_list = self.stage_3
                s_=2
                s_stage=3
            if stage_name == "stage_4":
                stage_list = self.stage_4
                s_=2
                s_stage=4
            if stage_name == "stage_5":
                stage_list = self.stage_5
                s_=2
                s_stage=5
            index = 0
            for layer_name in stage_list["structure_list"].split():
                #print(index)
                #print(layer_name)
                filter__ = stage_list[layer_name].split()
                filter_ = [int(filter__[i]) for i in range(len(filter__))]
                #self.check_X_ = X
                X = self.update_layer(X,layer_name, filter_, s_stage, str(index),s_)
                #print(X)
                index += 1

        X = AveragePooling2D()(X)

        # output layer
        X = Flatten()(X)
        X = Dense(1000, activation='softmax', name='fc', kernel_initializer=glorot_uniform(seed=0))(X)

        # Create model
        model = Model(inputs=X_input, outputs=X, name='ResNet50')

        self.model_architecture = model

        return model

    def prune_network_stack(self,structure_list):

        return self.stack_architecture(structure_list, input_shape=(64, 64, 3))

    def ResNet50_stack(self):
        structure_list = ["stage_2","stage_3","stage_4","stage_5"]

        return self.stack_architecture(structure_list, input_shape=(64, 64, 3))


    def sample_test_net(self, input_shape=(64, 64, 3), classes=1000):
        X_input = Input(input_shape)

        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)

        # Stage 1
        X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_regularizer=regularizers.L2(1e-4),
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='bn_conv1')(X)
        X = Activation('relu')(X)

        X = Conv2D(64, (3, 3), strides=(2, 2), name='conv2', kernel_regularizer=regularizers.L2(1e-4),
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='bn_conv2')(X)
        X = Activation('relu')(X)

        X = Conv2D(64, (2, 2), strides=(2, 2), name='conv3', kernel_regularizer=regularizers.L2(1e-4),
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='bn_conv3')(X)
        X = Activation('relu')(X)

        X = AveragePooling2D()(X)

        # output layer
        X = Flatten()(X)
        X = Dense(classes, activation='softmax', name='fc', kernel_initializer=glorot_uniform(seed=0))(X)
        self.check_X = X

        # Create model
        model = Model(inputs=X_input, outputs=X, name='ResNet50')

        return model


    def ResNet50(self, input_shape=(64, 64, 3), classes=1000):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

        Returns:
        model -- a Model() instance in Keras
        """

        # Define the input as a tensor with shape input_shape
        X_input = Input(input_shape)

        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)

        # Stage 1
        X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_regularizer=regularizers.L2(1e-4),
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = self.convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a',s=1)
        X = self.identity_block(X, 3, [64, 64, 256], stage=2, block='b')
        X = self.identity_block(X, 3, [64, 64, 256], stage=2, block='c')



        # Stage 3
        X = self.convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a',s=2)
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='b')
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='c')
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='d')

        # Stage 4
        X = self.convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a',s=2)
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

        # Stage 5
        X = self.convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a',s=2)
        X = self.identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
        X = self.identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

        # AVGPOOL . Use "X = AveragePooling2D(...)(X)"
        X = AveragePooling2D()(X)

        # output layer
        X = Flatten()(X)
        X = Dense(classes, activation='softmax', name='fc', kernel_initializer=glorot_uniform(seed=0))(X)
        self.check_X = X

        # Create model
        model = Model(inputs=X_input, outputs=X, name='ResNet50')

        return model
