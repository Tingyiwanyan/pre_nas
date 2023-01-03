from data_process import data_preprocess as dp
from lib import network_construct as nc
from lib import ntk_cal
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from sklearn.preprocessing import LabelBinarizer,LabelEncoder
from tensorflow.keras.metrics import categorical_accuracy,top_k_categorical_accuracy
import timeit
#from ntk_cal import ntk_compute

file_path_train_64 = '/home/tingyi/Downloads/img_net_subset/Imagenet64_train/train_data_batch_'

file_path_val_64 = '/home/tingyi/Downloads/img_net_subset/Imagenet64_val/val_data'


class train_network():
    def __init__(self):
        self.x_train = []
        self.y_train = []
        self.lb = LabelBinarizer()
        self.epoch = 100
        self.batch_size = 256
        #self.x_train, self.y_train = dp.load_data(file_path_train_64+str(1))
        for i in range(10):
            print(i)
            x_train, y_train = dp.load_data(file_path_train_64+str(i+1))
            #x_train = x_train / 255
            y_train = list(np.array(y_train) - 1)
            y_train = self.lb.fit_transform(y_train)
            self.x_train.append(x_train)
            self.y_train.append(y_train)
        self.x_val, self.y_val = dp.load_data(file_path_val_64)
        self.x_val = self.x_val

        self.y_val = list(np.array(self.y_val)-1)
        self.y_val = self.lb.fit_transform(self.y_val)

        self.length_train = self.x_train[0].shape[0]*10
        self.steps = self.length_train // self.batch_size
        #self.lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
            #initial_learning_rate=0.003, decay_steps=self.steps)

        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.03,
            decay_steps=self.steps,
            decay_rate=0.9)

        #data_set = tf.data.Dataset.from_tensor_slices((your_converted_listX, your_converted_listY))

        self.val_data = tf.data.Dataset.from_tensor_slices(
            (self.x_val, self.y_val))
        self.val_data = self.val_data.shuffle(buffer_size=1024, seed=4).batch(256)

        #self.steps = self.length_train // self.batch_size
        #self.lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
         #   initial_learning_rate=0.0003, decay_steps=self.steps)

    def top_5_accuracy(self, y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=5)

    def model_resnet50(self):
        self.nc_model = nc.network_construction()

        self.model = self.nc_model.ResNet50(input_shape=(64, 64, 3), classes=1000)
        #self.model = self.nc_model.sample_test_net(input_shape=(64, 64, 3), classes=1000)
        #self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', self.top_5_accuracy])

        #self.model.compile(loss='categorical_crossentropy',
                      #optimizer=K.optimizers.RMSprop(lr=2e-5),
                      #metrics=['accuracy'])

    def train_plain_supervised(self):
        self.top1_step = []
        self.top1_epoch = []
        self.top5_step = []
        self.top5_epoch = []
        self.top1_std_step = []
        self.top1_std_epoch = []
        self.top5_std_step = []
        self.top5_std_epoch = []
        self.loss_track = []
        self.epoch_run_time = []
        self.cce = tf.keras.losses.CategoricalCrossentropy()

        for epoch in range(self.epoch):
            print("\nStart of epoch %d" % (epoch,))
            predict_val = self.model.predict(self.x_val)
            self.check_predict_val = predict_val
            top_1_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=1)
            top_1_acc.update_state(self.y_val,predict_val)
            print("val_top1_acc:")
            print(top_1_acc.result().numpy())
            self.top1_epoch.append(top_1_acc.result().numpy())

            top_5_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
            top_5_acc.update_state(self.y_val,predict_val)
            print("val_top5_acc")
            print(top_5_acc.result().numpy())
            self.top5_epoch.append(top_5_acc.result().numpy())

            start = timeit.default_timer()
            self.step_plus = self.x_train[0].shape[0]//self.batch_size
            for ii in range(len(self.x_train)):
                print("in")
                print(ii)
                print("dataset")
                self.train_data = tf.data.Dataset.from_tensor_slices(
                    (self.x_train[ii], self.y_train[ii]))
                self.train_data = self.train_data.shuffle(buffer_size=1024, seed=4).batch(256)

                for step, (x_batch_train, y_batch_train) in enumerate(self.train_data):
                    x_batch_train = x_batch_train / 255
                    step = step+ii*self.step_plus
                    self.check_x_batch = x_batch_train
                    self.check_label = y_batch_train
                    with tf.GradientTape() as tape:
                        prediction = self.model(x_batch_train)
                        self.check_prediction = prediction
                        loss = self.cce(y_batch_train, prediction)
                        self.check_prediction = prediction

                    gradients = \
                        tape.gradient(loss,
                                      self.model.trainable_variables)
                    optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule,momentum=0.9)

                    optimizer.apply_gradients(zip(gradients,
                                                  self.model.trainable_variables))

                    if step % 100 == 0:
                        print("Training loss(for one batch) at step %d: %.4f"
                              % (step, float(loss)))
                        print("seen so far: %s samples" % ((step + 1) * self.batch_size))

                        predict_val = self.model.predict(x_batch_train)

                        top_1_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=1)
                        top_1_acc.update_state(y_batch_train,predict_val)
                        print("train_top1_acc:")
                        print(top_1_acc.result().numpy())
                        self.top1_step.append(top_1_acc.result().numpy())

                        top_5_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
                        top_5_acc.update_state(y_batch_train,predict_val)
                        print("train_top5_acc")
                        print(top_5_acc.result().numpy())
                        self.top5_step.append(top_5_acc.result().numpy())

                        self.loss_track.append(loss.numpy())
                        print("loss")
                        print(loss.numpy())

            stop = timeit.default_timer()
            self.epoch_run_time.append(stop-start)

        self.model.save("plain_supervised_resnet50")

    def save_exp_data(self,name):
        self.model.save(name)


if __name__ == '__main__':

    train_net = train_network()
    train_net.model_resnet50()
    k = train_net.nc_model.prune_network_stack(["stage_2","stage_3","stage_4","stage_5"])
    ntk_computation = ntk_cal.ntk_compute(train_net)