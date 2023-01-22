import numpy as np
import tensorflow as tf
from keras import backend as B
import math
from tensorflow.keras.layers import Input, Add, Dense, Dropout, \
    Activation, ZeroPadding2D, BatchNormalization, Flatten, \
    Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,GlobalAveragePooling2D,Concatenate, ReLU,\
    LeakyReLU,Reshape, Lambda
from tensorflow.keras.models import Sequential, load_model, Model

class ntk_compute():
    def __init__(self, model_init):
        self.model_init = model_init
        self.batch_size = 128
        self.seed = 42
        self.input_shape = (64, 64, 3)  ##desired shape of the image for resizing purposes
        self.val_sample = 0.1
        self.dummy_input = np.zeros((1,64,64,3))
        #self.ntk
        #self.strategy = tf.distribute.MirroredStrategy()


    def get_layer_compute_output(self, input, layer_num):
        #with self.strategy.scope():
        previous_layer_output = self.get_layer_output(input, layer_num-1)
        strides = self.model_init.model.layers[layer_num].get_config()['strides'][0]
        pad = self.model_init.model.layers[layer_num].get_config()['padding']
        w = self.model_init.model.layers[layer_num].get_weights()[0]
        b = self.model_init.model.layers[layer_num].get_weights()[1]
        self.check_previous_layer_output = previous_layer_output

        self.direct_output = self.get_layer_output(input, layer_num)
        self.computed_output = self.cnn_layer_compute_output(w, b, previous_layer_output, strides, pad)

    def get_maxpool_layer_compute_output(self, input, layer_num):
        previous_layer_output = self.get_layer_output(input, layer_num - 1)
        strides = self.model_init.model.layers[layer_num].get_config()['strides'][0]
        pad = self.model_init.model.layers[layer_num].get_config()['padding']
        pool_size = self.model_init.model.layers[layer_num].get_config()['pool_size'][0]
        self.check_previous_layer_output = previous_layer_output

        self.get_mask_max_pool(pool_size, previous_layer_output, strides, pad)




    def get_layer_output(self, input, layer_num):
        """
        return the layer values for specific layer
        """
        get_output = B.function([self.model_init.model.layers[0].input],
                                 [self.model_init.model.layers[layer_num].output])

        layer_output = get_output([input])
        return np.array(layer_output)[0]

    def cnn_layer_compute_output(self, w, b, input, strides, pad):
        """
        test function on validate compute output correctness
        """
        kernel_size = w.shape[0]
        input_shape = input.shape
        single_direction_size = input.shape[1]
        input = tf.reshape(input, [input_shape[0], input_shape[1] * input_shape[2], input_shape[3]])
        #input = tf.expand_dims(input, axis=-2)
        #input = tf.broadcast_to(input, [input_shape[0], input_shape[1] * input_shape[2], w.shape[-2], w.shape[-1]])
        if pad == "same":
            #if not kernel_size == 1:
                #if not kernel_size % 2 == 0:
            padding_size = (kernel_size - 1) / 2

            padding = tf.constant([0,0],[padding_size,padding_size],[padding_size,padding_size],[0,0])
            input = tf.pad(input,padding,"CONSTANT")

        w_window_start = list(range(0, single_direction_size-kernel_size, strides))
        w_window_start_whole = []
        #w_window_start_whole.append(w_window_start)
        for i in w_window_start:
            w_window_start_ = list(np.array(w_window_start)+i*single_direction_size)
            w_window_start_whole.append(w_window_start_)

        w_window_start_whole = np.array(w_window_start_whole)
        w_window_start_whole = \
            w_window_start_whole.reshape(1,w_window_start_whole.shape[0]*w_window_start_whole.shape[1])[0]
        self.check_w_window_start_whole = w_window_start_whole
        input = tf.expand_dims(input,axis=1)
        input = tf.broadcast_to(input,[input_shape[0],w_window_start_whole.shape[0],input.shape[2],input.shape[3]])
        self.check_input = input
        conv_field_mask = np.zeros((w_window_start_whole.shape[0], input.shape[2]))
        for i in range(w_window_start_whole.shape[0]):
            if kernel_size == 1:
                conv_field_mask[i,w_window_start_whole[i]:w_window_start_whole[i]+kernel_size] = 1
            else:
                for j in range(kernel_size):
                    # conv_field_mask[i, w_window_start_whole[i]:w_window_start_whole[i] + kernel_size] = 1
                    conv_field_mask[i, w_window_start_whole[i] + j * single_direction_size:
                                       w_window_start_whole[i] + j * single_direction_size + kernel_size] = 1
        conv_field_mask = tf.convert_to_tensor(conv_field_mask)
        conv_field_mask = tf.expand_dims(conv_field_mask,axis=0)
        conv_field_mask = tf.expand_dims(conv_field_mask,axis=-1)
        self.check_conv_field_mask = conv_field_mask
        conv_field_mask = tf.broadcast_to(conv_field_mask,
                                          [input.shape[0],input.shape[1],input.shape[2],input.shape[3]])

        #self.check_conv_field_mask = conv_field_mask
        #self.check_input = input

        input = tf.cast(input, tf.float32)
        #conv_field_mask = tf.cast(conv_field_mask, tf.float32)
        input = tf.transpose(input,[0,1,3,2])
        #conv_output_initial = tf.multiply(input,conv_field_mask)
        #self.check_conv_output_inital = conv_output_initial
        mask = tf.greater(conv_field_mask,0)
        mask = tf.transpose(mask,[0,1,3,2])
        self.check_mask = mask
        conv_output_mask = tf.boolean_mask(input,mask)
        conv_output_mask = tf.reshape(conv_output_mask,
                                      [input.shape[0],input.shape[1],input.shape[2],kernel_size*kernel_size])
        conv_output_mask = tf.transpose(conv_output_mask,[0,1,3,2])
        self.check_conv_output_mask = conv_output_mask

        w = tf.reshape(w,[w.shape[0]*w.shape[1],w.shape[2],w.shape[3]])
        #self.check_w_init = w
        w = tf.expand_dims(w, axis=0)
        w = tf.expand_dims(w, axis=0)
        w = tf.broadcast_to(w, [input.shape[0], input.shape[1], w.shape[-3],w.shape[-2], w.shape[-1]])
        #self.check_input_late = input
        self.check_w = w
        conv_output_mask_final = tf.expand_dims(conv_output_mask,axis=-1)
        conv_output_mask_final = tf.broadcast_to(conv_output_mask_final,
                                            [input.shape[0], input.shape[1], w.shape[2],w.shape[-2], w.shape[-1]])

        self.check_conv_output_final = conv_output_mask_final

        conv_output = tf.multiply(conv_output_mask_final,w)

        conv_output = tf.reduce_sum(conv_output,axis=2)
        conv_output = tf.reduce_sum(conv_output,axis=2)

        b = tf.expand_dims(b, axis=0)
        b = tf.expand_dims(b, axis=0)
        b = tf.broadcast_to(b, [conv_output.shape[0],conv_output.shape[1],conv_output.shape[2]])
        conv_output = tf.math.add(conv_output,b)
        self.check_conv_output = conv_output

        return conv_output#, mask[0,:,0,:], conv_parameters

    def get_mask_max_pool(self, pool_size, input, strides, pad):
        """
        return max pool layer mask
        """
        kernel_size = pool_size
        input_shape = input.shape
        single_direction_size = input.shape[1]
        input = tf.reshape(input, [input_shape[0], input_shape[1] * input_shape[2], input_shape[3]])
        # input = tf.expand_dims(input, axis=-2)
        # input = tf.broadcast_to(input, [input_shape[0], input_shape[1] * input_shape[2], w.shape[-2], w.shape[-1]])
        if pad == "same":
            # if not kernel_size == 1:
            # if not kernel_size % 2 == 0:
            padding_size = (kernel_size - 1) / 2

            padding = tf.constant([0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0])
            input = tf.pad(input, padding, "CONSTANT")

        w_window_start = list(range(0, single_direction_size-kernel_size+1, strides))
        w_window_start_whole = []
        # w_window_start_whole.append(w_window_start)
        for i in w_window_start:
            w_window_start_ = list(np.array(w_window_start) + i * single_direction_size)
            w_window_start_whole.append(w_window_start_)

        w_window_start_whole = np.array(w_window_start_whole)
        w_window_start_whole = \
            w_window_start_whole.reshape(1, w_window_start_whole.shape[0] * w_window_start_whole.shape[1])[0]
        self.check_w_window_start_whole = w_window_start_whole
        input = tf.expand_dims(input, axis=1)
        input = tf.broadcast_to(input, [input_shape[0], w_window_start_whole.shape[0], input.shape[2], input.shape[3]])
        conv_field_mask = np.ones((w_window_start_whole.shape[0], input_shape[2])) * -math.inf
        for i in range(w_window_start_whole.shape[0]):
            if kernel_size == 1:
                conv_field_mask[i, w_window_start_whole[i]:w_window_start_whole[i] + kernel_size] = 1
            else:
                for j in range(kernel_size):
                    #conv_field_mask[i, w_window_start_whole[i]:w_window_start_whole[i] + kernel_size] = 1
                    conv_field_mask[i, w_window_start_whole[i] + j*single_direction_size:
                                       w_window_start_whole[i] + j*single_direction_size + kernel_size] = 1
        conv_field_mask = tf.convert_to_tensor(conv_field_mask)
        conv_field_mask = tf.expand_dims(conv_field_mask, axis=0)
        conv_field_mask = tf.expand_dims(conv_field_mask, axis=-1)
        conv_field_mask = tf.broadcast_to(conv_field_mask,
                                          [input.shape[0], input.shape[1], input.shape[2], input.shape[3]])

        conv_field_mask = tf.cast(conv_field_mask, tf.float32)

        input = tf.cast(input, tf.float32)
        max_pool_mask = tf.multiply(conv_field_mask, input)

        self.check_conv_field_mask = conv_field_mask
        self.check_max_pool_mask = max_pool_mask

        max_pool_mask_index = tf.math.argmax(max_pool_mask, axis=2)

        self.check_max_pool_mask_index = max_pool_mask_index

        input = tf.cast(input, tf.float32)
        mask_output = tf.math.multiply(conv_field_mask, input)
        self.check_mask_output = mask_output
        max_pool_mask_index = tf.reshape(max_pool_mask_index,
                                         [max_pool_mask_index.shape[0]*
                                          max_pool_mask_index.shape[1]*max_pool_mask_index.shape[2]])
        mask_output = tf.transpose(mask_output,[0,1,3,2])
        mask_output = tf.reshape(mask_output,[mask_output.shape[0]*
                                              mask_output.shape[1]*mask_output.shape[2],mask_output.shape[3]])
        counting_index = tf.range(max_pool_mask_index.shape[0])
        counting_index = tf.cast(counting_index,tf.int64)
        max_pool_mask_index = tf.stack([counting_index,max_pool_mask_index])
        max_pool_mask_index = tf.transpose(max_pool_mask_index,[1,0])
        self.check_max_index = max_pool_mask_index

        output = tf.gather_nd(mask_output,list(max_pool_mask_index))
        self.check_max_pool_output = output

    def get_mask_single_layer(self, w, input, strides, pad):
        """
        get single layer mask output, including max-pooling and average pooling
        """
        kernel_size = w.shape[0]
        input_shape = input.shape
        single_direction_size = input.shape[1]
        input = tf.reshape(input, [input_shape[0], input_shape[1] * input_shape[2], input_shape[3]])
        # input = tf.expand_dims(input, axis=-2)
        # input = tf.broadcast_to(input, [input_shape[0], input_shape[1] * input_shape[2], w.shape[-2], w.shape[-1]])
        if pad == "same":
            # if not kernel_size == 1:
            # if not kernel_size % 2 == 0:
            padding_size = (kernel_size - 1) / 2

            padding = tf.constant([0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0])
            input = tf.pad(input, padding, "CONSTANT")

        w_window_start = list(range(0, single_direction_size-kernel_size+1, strides))
        w_window_start_whole = []
        # w_window_start_whole.append(w_window_start)
        for i in w_window_start:
            w_window_start_ = list(np.array(w_window_start) + i * single_direction_size)
            w_window_start_whole.append(w_window_start_)

        w_window_start_whole = np.array(w_window_start_whole)
        w_window_start_whole = \
            w_window_start_whole.reshape(1, w_window_start_whole.shape[0] * w_window_start_whole.shape[1])[0]
        self.check_w_window_start_whole = w_window_start_whole
        input = tf.expand_dims(input, axis=1)
        input = tf.broadcast_to(input, [input_shape[0], w_window_start_whole.shape[0], input.shape[2], input.shape[3]])
        conv_field_mask = np.zeros((w_window_start_whole.shape[0], input.shape[2]))
        for i in range(w_window_start_whole.shape[0]):
            if kernel_size == 1:
                conv_field_mask[i, w_window_start_whole[i]:w_window_start_whole[i] + kernel_size] = 1
            else:
                for j in range(kernel_size):
                    # conv_field_mask[i, w_window_start_whole[i]:w_window_start_whole[i] + kernel_size] = 1
                    conv_field_mask[i, w_window_start_whole[i] + j * single_direction_size:
                                       w_window_start_whole[i] + j * single_direction_size + kernel_size] = 1
        conv_field_mask = tf.convert_to_tensor(conv_field_mask)
        conv_field_mask = tf.expand_dims(conv_field_mask, axis=0)
        conv_field_mask = tf.expand_dims(conv_field_mask, axis=-1)
        conv_field_mask = tf.broadcast_to(conv_field_mask,
                                          [input.shape[0], input.shape[1], input.shape[2], input.shape[3]])

        #input = tf.cast(input,tf.float32)
        conv_field_mask = tf.cast(conv_field_mask,tf.float32)
        #max_pool_mask = tf.multiply(conv_field_mask, input)

        #self.check_conv_field_mask = conv_field_mask
        #self.check_max_pool_mask = max_pool_mask

        #max_pool_mask_index = tf.math.argmax(max_pool_mask,axis=2)

        #self.check_max_pool_mask_index = max_pool_mask_index

        mask = tf.greater(conv_field_mask, 0)
        mask = tf.transpose(mask, [0, 1, 3, 2])
        mask_previous_layer = mask[0,:,0,:]
        self.check_previous_mask_layer = mask_previous_layer
        self.check_mask = mask
        input_reshape = input

        return mask, input_reshape, kernel_size, mask_previous_layer

    def get_derivative_single_filter(self, input, mask, kernel_size,alpha_filter_index,output_size):

        self.check_input = input

        input = tf.cast(input, tf.float32)
        input = tf.transpose(input, [0, 1, 3, 2])
        conv_output_mask = tf.boolean_mask(input, mask)
        conv_output_mask = tf.reshape(conv_output_mask,
                                      [input.shape[0], input.shape[1], input.shape[2], kernel_size * kernel_size])
        conv_output_mask = tf.transpose(conv_output_mask, [0, 1, 3, 2])
        self.check_conv_output_mask = conv_output_mask

        """
        return single layer derivative w.r.t its parameters: filter_size_l * filter_size_(l+1)
        """
        #conv_parameters_w = conv_output_mask[:,:,:,filter_num]

        conv_parameters_w = \
            tf.reshape(conv_output_mask,[conv_output_mask.shape[0],conv_output_mask.shape[1],
                                          conv_output_mask.shape[2]*conv_output_mask.shape[3]])

        self.check_conv_parameters_w = conv_parameters_w

        """
        the sub-derivative for b is always 1, so concatenate all 1 tensor to w, the difference of b will be reflected
        on the output difference
        """
        # conv_parameters_b = tf.ones([input.shape[0],input.shape[1],w.shape[-1]])
        conv_parameters_b = np.zeros([input.shape[0], conv_parameters_w.shape[1], 1])

        conv_parameters = tf.concat([conv_parameters_w,conv_parameters_b],axis=-1)

        self.check_conv_parameters = conv_parameters

        conv_parameters_output = np.zeros((conv_parameters.shape[0],
                                          conv_parameters.shape[1]*output_size,conv_parameters.shape[2]))
        conv_parameters_output[:,
        conv_parameters.shape[1]*alpha_filter_index:conv_parameters.shape[1]*(alpha_filter_index+1),:]=conv_parameters

        conv_parameters_output = tf.convert_to_tensor(conv_parameters_output)

        return conv_parameters_output


    def cnn_layer_derivative(self, inputs, layer_num, alpha_filter_index, is_bottom=False):
        """
        return single cnn layer derivative
        """
        inputs = inputs / 255
        previous_layer_output = self.get_layer_output(inputs, layer_num - 1)
        strides = self.model_init.model.layers[layer_num].get_config()['strides'][0]
        pad = self.model_init.model.layers[layer_num].get_config()['padding']
        w = self.model_init.model.layers[layer_num].get_weights()[0]
        #b = self.model_init.model.layers[layer_num].get_weights()[1]
        mask, input_reshape, kernel_size, mask_previous_layer = \
            self.get_mask_single_layer(w, previous_layer_output, strides, pad)
        if is_bottom:
            conv_parameters = self.get_derivative_single_filter(input_reshape, mask, kernel_size,
                                                                alpha_filter_index,w.shape[-1])
            return conv_parameters
        else:
            return mask_previous_layer, w


    def single_recur_ntk_model(self, recur_shape, curr_layer_shape, curr_layer_cnn_num, 
        bottem_layer_alpha_filter_index):

        recur_result = Input(recur_shape)
        curr_layer_output = Input(curr_layer_shape)
        mask_previous_layer, w = self.cnn_layer_derivative(self.dummy_input, curr_layer_cnn_num,
                                                               bottem_layer_alpha_filter_index)
        w_shape = w.shape
        w = tf.reshape(w, [w.shape[0] * w.shape[1] * w.shape[2], w.shape[3]])
        mask_previous_layer_ = []
        for i in range(w_shape[-2]):
            #mask_previous_layer = tf.concat([mask_previous_layer,mask_previous_layer],axis=1)
            mask_previous_layer_.append(mask_previous_layer)
        mask_previous_layer = tf.concat(mask_previous_layer_,axis=1)
        self.check_mask_previous_layer_new = mask_previous_layer
        recur_result = tf.cast(tf.transpose(recur_result,[0,2,1]),tf.float32)
        input_shape = tf.shape(recur_result)
        recur_result = tf.expand_dims(recur_result,axis=1)
        recur_result = tf.broadcast_to(recur_result,[input_shape[0],mask_previous_layer.shape[0],input_shape[1],input_shape[2]])
        self.check_recur_result = recur_result
        #recur_result = tf.cast(tf.transpose(recur_result,[1,0]),tf.float32)
        #mask_previous_layer = tf.transpose(mask_previous_layer,[0,1,3,2])
        single_filter_output_derivative_ = []
        filter_output_derivative = []
        mask_single = mask_previous_layer
        mask_single = tf.expand_dims(mask_single,axis=1)
        mask_single = tf.expand_dims(mask_single,axis=0)
        mask_single = tf.broadcast_to(mask_single, [tf.shape(recur_result)[0],mask_previous_layer.shape[0],
            recur_shape[1],mask_single.shape[-1]])
        self.check_mask_single = mask_single

        single_filter_output_derivative = tf.boolean_mask(recur_result, mask_single)
        self.check_single_output_derivative_ = single_filter_output_derivative
        for i in range(w.shape[-1]):
            print("construct")
            print(i)
            #for j in range(mask_previous_layer.shape[0]):
            w_single = w[:,i]
            w_single = tf.expand_dims(w_single,axis=0)
            w_single = tf.expand_dims(w_single,axis=0)
            w_single = tf.expand_dims(w_single,axis=0)
            w_single = tf.broadcast_to(w_single,[tf.shape(recur_result)[0],mask_previous_layer.shape[0],
                recur_shape[1],w_single.shape[-1]])
            #mask_single = tf.expand_dims(mask_single,axis=0)
            single_filter_output_derivative = tf.reshape(single_filter_output_derivative,tf.shape(w_single))
            self.check_w_single = w_single
            self.check_single_filter_output_derivative = single_filter_output_derivative
            single_filter_output_derivative = tf.multiply(w_single,single_filter_output_derivative)

            single_filter_output_derivative = tf.reduce_sum(single_filter_output_derivative,axis=-1)
            #single_filter_output_derivative_.append(single_filter_output_derivative)
            #self.check_single_filter_output_derivative_ = single_filter_output_derivative_
            #single_output_derivative = tf.stack(single_filter_output_derivative_,axis=1)
            #single_filter_output_derivative_ = []

            self.check_single_output_derivative = single_filter_output_derivative
            filter_output_derivative.append(single_filter_output_derivative)

        output_derivative = tf.concat(filter_output_derivative,axis=1)

        self.check_output_derivative = output_derivative

        curr_layer_output = tf.reshape(curr_layer_output,
                                       [tf.shape(curr_layer_output)[0], curr_layer_output.shape[1] *
                                        curr_layer_output.shape[2] *
                                        curr_layer_output.shape[-1]])
        self.check_curr_layer_output = curr_layer_output
        curr_layer_output = tf.expand_dims(curr_layer_output, axis=2)
        curr_layer_output = tf.broadcast_to(curr_layer_output, [tf.shape(output_derivative)[0],
            output_derivative.shape[1],output_derivative.shape[2]])

        output_derivative = tf.multiply(output_derivative, curr_layer_output)

        #mask_previous_layer_index = list(tf.where(mask_previous_layer))

        #output_derivative = tf.gather_nd(recur_result,mask_previous_layer_index)
        #output_derivative = tf.reduce_sum(output_derivative,axis=2)

        

        model = Model(inputs=[recur_result, curr_layer_output], outputs=output_derivative)
        return model

    #def recur_build_derivative_model(self, top_layer_num, bottom_layer_num):


    def recur_layer_derivative(self, input, curr_layer_num, bottom_layer_num, bottem_layer_alpha_filter_index):
        #w_bottom_shape = self.model_init.model.layers[bottom_layer_num].get_weights()[0].shape
        #w_bottom_shape = w_bottom_shape[0]*w_bottom_shape[1]*w_bottom_shape[2]*w_bottom_shape[3]
        curr_layer = self.model_init.model.layers[curr_layer_num]
        layer_output_type = curr_layer.output.name.split("/")[1].split(":")[0]
        while not layer_output_type == "Relu":
            curr_layer_num = curr_layer_num - 1
            curr_layer = self.model_init.model.layers[curr_layer_num]
            layer_output_type = curr_layer.output.name.split("/")[1].split(":")[0]

        curr_layer_output = self.get_layer_output(input, curr_layer_num)
        #curr_layer_output_shape = self.model_init.model.layers[curr_layer_num].output.shape
        curr_layer_output = tf.reshape(curr_layer_output,
                                       [curr_layer_output.shape[0], curr_layer_output.shape[1] *
                                        curr_layer_output.shape[2] *
                                        curr_layer_output.shape[3]])
        curr_layer_output = tf.expand_dims(curr_layer_output, axis=2)
        self.check_curr_layer_output = curr_layer_output

        self.check_curr_layer_num = curr_layer_num
        #self.check_curr_layer_output = curr_layer_output
        curr_layer_cnn_num = curr_layer_num
        #curr_layer_cnn = self.model_init.model.layers[curr_layer_cnn_num]
        while not layer_output_type == "BiasAdd":
            curr_layer_cnn_num = curr_layer_cnn_num - 1
            curr_layer_cnn = self.model_init.model.layers[curr_layer_cnn_num]
            layer_output_type = curr_layer_cnn.output.name.split("/")[1].split(":")[0]

        self.check_curr_layer_cnn_num = curr_layer_cnn_num
        #while not curr_layer_cnn_num == bottom_layer_num:
        if not curr_layer_cnn_num == bottom_layer_num:
            """
            reverse the recursive order so as to save space
            """
            print("Im in iterative")
            print(curr_layer_output.shape)
            recur_result = self.recur_layer_derivative(input, curr_layer_num-1,bottom_layer_num,
                                                       bottem_layer_alpha_filter_index)

            #curr_layer_output = self.model_init.model.layers[curr_layer_num].output
            #curr_layer_output = tf.reshape(curr_layer_output,
                                           #[curr_layer_output.shape[0], curr_layer_output.shape[1] *
                                            #curr_layer_output.shape[2] *
                                            #curr_layer_output.shape[3]])
            #curr_layer_output_shape = self.model_init.model.layers[curr_layer_num].output.shape

            mask_previous_layer, w = self.cnn_layer_derivative(input, curr_layer_cnn_num,
                                                               bottem_layer_alpha_filter_index)
            w_shape = w.shape
            w = tf.reshape(w, [w.shape[0] * w.shape[1] * w.shape[2], w.shape[3]])
            #mask_previous_layer = tf.expand_dmis(mask_previous_layer, axis=0)
            #mask_previous_layer = tf.expand_dims(mask_previous_layer, axis=-1)
            #mask_previous_layer = tf.broadcast_to(mask_previous_layer, [input.shape[0], mask_previous_layer.shape[1],
                                                  #mask_previous_layer.shape[2], w.shape[1]])
            # mask_previous_layer = tf.reshape(mask_previous_layer,[input.shape[0]],mask_previous_layer.shape[1],
            # mask_previous_layer.shape[2]*w.shape[1])
            mask_previous_layer_ = []
            for i in range(w_shape[-2]):
                #mask_previous_layer = tf.concat([mask_previous_layer,mask_previous_layer],axis=1)
                mask_previous_layer_.append(mask_previous_layer)
            mask_previous_layer = tf.concat(mask_previous_layer_,axis=1)

            self.check_mask_previous_layer = mask_previous_layer
            print("success create mask")

            #recur_result = tf.expand_dims(recur_result, axis=1)
            #recur_result = tf.broadcast_to(recur_result,(recur_result.shape[0],
                                                         #curr_layer_output_shape[1]*curr_layer_shape[2],
                                                         #recur_result.shape[2],
                                                         #recur_result.shape[3]))
            self.check_recur_result = recur_result
            recur_result = tf.cast(tf.transpose(recur_result,[0,2,1]),tf.float32)
            #mask_previous_layer = tf.transpose(mask_previous_layer,[0,1,3,2])
            single_filter_output_derivative_ = []
            filter_output_derivative = []
            for i in range(w.shape[-1]):
                for j in range(mask_previous_layer.shape[0]):
                    w_single = w[:,i]
                    w_single = tf.expand_dims(w_single,axis=0)
                    w_single = tf.expand_dims(w_single,axis=0)
                    w_single = tf.broadcast_to(w_single,[recur_result.shape[0],
                                               recur_result.shape[1],w_single.shape[-1]])
                    mask_single = mask_previous_layer[j]
                    mask_single = tf.expand_dims(mask_single,axis=0)
                    mask_single = tf.expand_dims(mask_single,axis=0)
                    mask_single = tf.broadcast_to(mask_single, [recur_result.shape[0],recur_result.shape[1],
                                                  recur_result.shape[2]])
                    single_filter_output_derivative = tf.boolean_mask(recur_result, mask_single)
                    single_filter_output_derivative = tf.reshape(single_filter_output_derivative,
                                                                 [recur_result.shape[0],
                                                                  recur_result.shape[1],w_single.shape[-1]])
                    self.check_w_single = w_single
                    self.check_single_filter_output_derivative = single_filter_output_derivative
                    single_filter_output_derivative = w_single*single_filter_output_derivative

                    single_filter_output_derivative = tf.reduce_sum(single_filter_output_derivative,axis=-1)
                    single_filter_output_derivative_.append(single_filter_output_derivative)
                self.check_single_filter_output_derivative_ = single_filter_output_derivative_
                single_output_derivative = tf.stack(single_filter_output_derivative_,axis=1)
                single_filter_output_derivative_ = []

                self.check_single_output_derivative = single_output_derivative
                filter_output_derivative.append(single_output_derivative)

            output_derivative = tf.concat(filter_output_derivative,axis=1)
            output_derivative = output_derivative * curr_layer_output

            #mask_previous_layer_index = list(tf.where(mask_previous_layer))

            #output_derivative = tf.gather_nd(recur_result,mask_previous_layer_index)
            #output_derivative = tf.reduce_sum(output_derivative,axis=2)

            self.check_output_derivative = output_derivative
            return output_derivative
        else:
            print("I'm in base")
            print(curr_layer_output.shape)
            return tf.cast(curr_layer_output,tf.float64) * self.cnn_layer_derivative(input, curr_layer_cnn_num,
                                                               bottem_layer_alpha_filter_index, is_bottom=True)


    def resnet_derivative(self, input, desire_layer_num, bottom_layer_num, bottem_layer_alpha_filter_index):
        """
        compute network derivative with sepecific filter
        """
        single_filter_whole_derivative = \
            self.recur_layer_derivative(input, desire_layer_num, bottom_layer_num, bottem_layer_alpha_filter_index)

        self.check_single_filter_whole_derivative = single_filter_whole_derivative



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