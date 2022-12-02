import numpy as np
import tensorflow as tf
from keras import backend as B

class ntk_compute():
    def __init__(self, model_init):
        self.model_init = model_init
        self.batch_size = 128
        self.seed = 42
        self.input_shape = (64, 64, 3)  ##desired shape of the image for resizing purposes
        self.val_sample = 0.1

    def get_layer_output(self, input, layer_num):
        """
        return the layer values for specific layer
        """
        get_output = B.function([self.model_init.model_architecture.layers[0].input,
                                 self.model_init.model_architecture.layers[layer_num].output])

        layer_output = get_output([input])
        return layer_output

    def cnn_layer_compute_output(self, w, input, strides, pad):
        """
        test function on validate compute output correctness
        """
        kernel_size = w.shape[0]
        input_shape = input.shape
        input = tf.reshape(input, [input_shape[0], input_shape[1] * input_shape[2], input_shape[3]])
        #input = tf.expand_dims(input, axis=-2)
        #input = tf.broadcast_to(input, [input_shape[0], input_shape[1] * input_shape[2], w.shape[-2], w.shape[-1]])
        if pad == "same":
            if not kernel_size == 1:
                #if not kernel_size % 2 == 0:
                padding_size = (kernel_size - 1) / 2

                padding = tf.constant([0,0],[padding_size,padding_size],[padding_size,padding_size],[0,0])
                input = tf.pad(input,padding,"CONSTANT")

        single_direction_size = input.shape[0,1,0,0]
        w_window_start = list(range(0, single_direction_size, strides))
        w_window_start_whole = []
        w_window_start_whole.append(w_window_start)
        for i in w_window_start:
            w_window_start_ = list(np.array(w_window_start)+i*single_direction_size)
            w_window_start_whole.append(w_window_start_)

        w_window_start_whole = \
            np.array(w_window_start_whole).reshape(1,w_window_start_whole.shape[0]*w_window_start_whole.shape[0])[0]
        input = tf.expand_dims(input,axis=1)
        input = tf.broadcast_to(input,[input_shape[0],w_window_start_whole.shape[0],input.shape[2],input.shape[3]])
        conv_field_mask = np.array((w_window_start_whole.shape[0], input_shape[1]*input_shape[2]))
        for i in range(w_window_start_whole.shape[0]):
            conv_field_mask[i,w_window_start_whole[i]:w_window_start_whole[i]+kernel_size] = 1
            conv_field_mask[i,w_window_start_whole[i]+single_direction_size:
                              w_window_start_whole[i]+single_direction_size+kernel_size] = 1
        conv_field_mask = tf.convert_to_tensor(conv_field_mask)
        conv_field_mask = tf.expand_dims(conv_field_mask,axis=0)
        conv_field_mask = tf.expand_dims(conv_field_mask,axis=-1)
        conv_field_mask = tf.broadcast_to(conv_field_mask,
                                          [input.shape[0],input.shape[1],input.shape[2],input.shape[3]])

        conv_output_initual = tf.multiply(input,conv_field_mask)
        mask = tf.greater(conv_output_initual,0)
        conv_output_mask = tf.boolean_mask(conv_output_initual,mask)

        self.check_conv_output_mask = conv_output_mask
        w = tf.reshape(w,[w.shape[0]*w.shape[1],w.shape[2],w.shape[3]])
        w = tf.expand_dims(w, axis=0)
        w = tf.expand_dims(w, axis=0)
        w = tf.broadcast_to(w, [input_shape[0], input_shape[1], w.shape[2],w.shape[-2], w.shape[-1]])

        conv_output_mask_final = tf.expand_dims(conv_output_mask,axis=-1)
        conv_output_mask_final = tf.broadcast_to(conv_output_mask_final,
                                            [input_shape[0], input_shape[1], w.shape[2],w.shape[-2], w.shape[-1]])

        self.check_conv_output_final = conv_output_mask_final
        conv_output = tf.multiply(conv_output_mask_final,w)

        conv_output = tf.reduce_sum(conv_output,axis=2)

        self.check_conv_output = conv_output
        return conv_output





    def cnn_layer_derivative(self, w, input, output, strides, pad):
        """
        return single layer derivative, can be computed recursively
        """
        a = 3



    def full_connect_net_ntk(self, x, w, theta):
        x_extend = tf.expand_dims(x, 1)
        x_extend = tf.broadcast_to()

    #def cnn_derivative(self, input, w):


    def identity_block(self, X, f, filters, stage, block):
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
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value. You'll need this later to add back to the main path.
        X_shortcut = X

        # First component of main path
        X = Conv2D(filters=F1, kernel_regularizer=regularizers.L2(1e-4), kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path
        X = Conv2D(filters=F2, kernel_regularizer=regularizers.L2(1e-4), kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path
        X = Conv2D(filters=F3, kernel_regularizer=regularizers.L2(1e-4), kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X