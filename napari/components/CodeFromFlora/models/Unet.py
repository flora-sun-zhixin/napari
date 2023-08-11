# The Auto-encoder object

# Flora Sun, CIG, WUSTL, 2021
# TensorFlow v1

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
from tensorflow.keras import backend as K
tf.disable_v2_behavior()


class Unet(object):
    """
    data_kargs:
        nx, ny, (nz) ~ 2D/3D spatial size of the image
        ic ~ input data channel size
        oc ~ ground truth channel size

    net_kargs:
        layer_num ~ the number of layers (dynamically adapted to the architecture)
        filter_size ~ the size of the convolutional filter
        feature_root ~ the starting number of feature maps

    train_kargs:
        batch_size ~ the size of training batch
        valid_size ~ the size of valid batch
        learning_rate ~ could be a list of learning rate corresponding to different epoches
        epoches ~ number of epoches
        is_restore ~ True / False
        prediction_path ~ where to save predicted results. No saves if set to None. (also used to save validation)
        save_epoch ~ save model every save_epochs

    """

    def __init__(self,
                 data_kargs={'ic': 1, 'oc': 4,
                             'img_height': 172,
                             'img_width': 679,
                             'measurement_smooth': 100,
                             'loss_weights': [1, 5, 5, 5],
                             'loss_alpha': 0.75,
                             'loss_gamma': 3},
                 net_kargs={},
                 gpu_ratio=0.2):
        tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.DEBUG)

        # dictionary of key args
        self.data_kargs = data_kargs

        # placeholders
        self.x = tf.placeholder(tf.float32, shape=[None,
                                                   self.data_kargs["img_height"],
                                                   self.data_kargs["img_width"],
                                                   self.data_kargs['ic']])
        self.y = tf.placeholder(tf.float32, shape=[None,
                                                   self.data_kargs["img_height"],
                                                   self.data_kargs["img_width"],
                                                   self.data_kargs['oc']])
        self.lr = tf.placeholder(tf.float32)
 
        # config
        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = gpu_ratio

        # define the architecture
        self.xhat = self.net(**net_kargs)
        self.measurement_smooth = data_kargs['measurement_smooth']
        self.loss_weights = data_kargs['loss_weights']
        self.loss, self.miou = self._get_measure()

    def net(self):
        inputs = self.x
        
        # unet shape should be (16m)
        paddings = tf.constant([[0, 0], [2, 2], [4, 5], [0, 0]])
        padded = tf.pad(inputs, paddings, "CONSTANT")

        conv1 = tf.layers.conv2d(inputs=padded, filters=32, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        # conv1 = tf.layers.dropout(inputs=conv1, rate=0.1)
        conv1 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        p1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2, padding="valid")

        conv2 = tf.layers.conv2d(inputs=p1, filters=64, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        # conv2 = tf.layers.dropout(inputs=conv2, rate=0.1)
        conv2 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        p2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2, padding="valid")

        conv3 = tf.layers.conv2d(inputs=p2, filters=128, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        # conv3 = tf.layers.dropout(inputs=conv3, rate=0.1)
        conv3 = tf.layers.conv2d(inputs=conv3, filters=128, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        p3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=2, strides=2, padding="valid")

        conv4 = tf.layers.conv2d(inputs=p3, filters=256, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        # conv4 = tf.layers.dropout(inputs=conv4, rate=0.1)
        conv4 = tf.layers.conv2d(inputs=conv4, filters=256, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        p4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=2, strides=2, padding="valid")

        conv5 = tf.layers.conv2d(inputs=p4, filters=512, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        # conv5 = tf.layers.dropout(inputs=conv5, rate=0.1)
        conv5 = tf.layers.conv2d(inputs=conv5, filters=512, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())

        up1 = tf.layers.conv2d_transpose(inputs=conv5, filters=256, kernel_size=2, strides=2,
                                         padding="same", kernel_initializer=tf.initializers.glorot_uniform())
        up1 = tf.concat([conv4, up1], axis=3)
        conv6 = tf.layers.conv2d(inputs=up1, filters=256, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        # conv6 = tf.layers.dropout(inputs=conv6, rate=0.1)
        conv6 = tf.layers.conv2d(inputs=conv6, filters=256, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        
        up2 = tf.layers.conv2d_transpose(inputs=conv6, filters=128, kernel_size=2, strides=2,
                                         padding="same", kernel_initializer=tf.initializers.glorot_uniform())
        up2 = tf.concat([conv3, up2], axis=3)
        conv7 = tf.layers.conv2d(inputs=up2, filters=128, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        # conv7 = tf.layers.dropout(inputs=conv7, rate=0.1)
        conv7 = tf.layers.conv2d(inputs=conv7, filters=128, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        
        up3 = tf.layers.conv2d_transpose(inputs=conv7, filters=64, kernel_size=2, strides=2,
                                         padding="same", kernel_initializer=tf.initializers.glorot_uniform())
        up3 = tf.concat([conv2, up3], axis=3)
        conv8 = tf.layers.conv2d(inputs=up3, filters=64, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        # conv8 = tf.layers.dropout(inputs=conv8, rate=0.1)
        conv8 = tf.layers.conv2d(inputs=conv8, filters=64, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())

        up4 = tf.layers.conv2d_transpose(inputs=conv8, filters=32, kernel_size=2, strides=2,
                                         padding="same", kernel_initializer=tf.initializers.glorot_uniform())
        up4 = tf.concat([conv1, up4], axis=3)
        conv9 = tf.layers.conv2d(inputs=up4, filters=32, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        # conv9 = tf.layers.dropout(inputs=conv9, rate=0.1)
        conv9 = tf.layers.conv2d(inputs=conv9, filters=32, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        
        conv10 = tf.layers.conv2d(inputs=conv9, filters=self.data_kargs["oc"], kernel_size=1, strides=1,
                                  kernel_initializer=tf.initializers.glorot_uniform())
        softmax = tf.nn.sigmoid(conv10)#, axis=-1)
        
        return softmax[:, 2:-2, 4:-5, :]

    def predict(self, 
                model_path, 
                x_test):
        with tf.Session(config=self.config) as sess:
            # Initialize variables
            sess.run(tf.global_variables_initializer())
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            # set phase to False for every prediction
            prediction = sess.run(self.xhat, feed_dict={self.x: x_test})

        return prediction

    def save(self,
             sess,
             model_path):

        # saver = tf.train.Saver(
        #     var_list=[v for v in tf.global_variables(scope='MLP')])
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self,
                sess,
                model_path):
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        tf.logging.info("Model restored from file: %s" % model_path)

    def train(self, 
              output_path, 
              train_provider, 
              valid_provider,
              batch_size=20, 
              valid_size=20, 
              epochs=80, 
              learning_rate=0.001,
              is_restore=False, 
              prediction_path='predict', 
              save_epoch=10):

        abs_output_path, abs_prediction_path = self._path_checker(
            output_path, prediction_path, is_restore)

        # define the optimizer
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     self.optimizer = tf.train.AdamOptimizer(
        #         learning_rate=self.lr).minimize(self.loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        # optimizer = tf.compat.v1.train.MomentumOptimizer(
        #     learning_rate=self.lr, momentum=0.3, use_locking=False, name='Momentum',
        #     use_nesterov=True
        # )
        train_op = optimizer.minimize(self.loss)

        # create output path
        directory = os.path.join(abs_output_path, "final/")
        if not os.path.exists(directory):
            os.makedirs(directory)
        save_path = os.path.join(directory, "model")
        if epochs == 0:
            tf.logging.info('Parameter [epoch] is zero. Programm terminated.')
            quit()

        # tensorflow
        with tf.Session(config=self.config) as sess:

            # initialize the session
            sess.run(tf.global_variables_initializer())

            if is_restore:
                model = tf.train.get_checkpoint_state(abs_output_path)
                if model and model.model_checkpoint_path:
                    self.restore(sess, model.model_checkpoint_path)

            # initialize summary_writer
            summary_writer = tf.summary.FileWriter(
                abs_output_path, graph=sess.graph)
            tf.logging.info('Start Training')

            # tracking the model with the highest miou
            best = 0

            # main loop for training
            global_step = 1
            raw_iters = train_provider.file_count / batch_size
            iters_per_epoch = int(
                raw_iters) + 1 if raw_iters > int(raw_iters) else int(raw_iters)

            for epoch in range(epochs):
                # reshuffle the order of feeding data
                train_provider.reset()
                # select validation dataset (1 is dummy placeholder)
                # valid_x, valid_y = valid_provider(valid_size, iter)

                iter = 0
                while iter < iters_per_epoch:

                    # extract training data
                    batch_x, batch_y = train_provider(batch_size, iter)
                    
                    # Learning rate
                    if type(learning_rate) is np.ndarray:
                        lr = learning_rate[epoch]
                    elif type(learning_rate) is float:
                        lr = learning_rate
                    else:
                        tf.logging.info(
                            'Learning rate should be a list of double or a double scalar.')
                        quit()

                    # run backprop
                    _, loss, miou, img = sess.run([train_op, self.loss, self.miou, self.xhat],
                                                  feed_dict={self.x: batch_x,
                                                             self.y: batch_y,
                                                             self.lr: lr})
                    
                    if(iter % 80 == 0):
                        plt.figure(figsize=(15, 2))
                        plt.subplot(1, 3, 1)
                        plt.imshow(batch_x[0, :, :, 0], cmap="gray")
                        
                        plt.subplot(1, 3, 2)
                        plt.imshow(batch_y[0, :, :, 0], cmap="gray")
                        plt.text(10, 10, f"{np.min(batch_y[0, :, :, 0]), np.max(batch_y[0, :, :, 0])}", color="yellow")
                        plt.subplot(1, 3, 3)
                        plt.imshow(img[0, :, :, 0], cmap="gray")
                        plt.text(10, 10, f"{np.min(img[0, :, :, 0]), np.max(img[0, :, :, 0])}", color="yellow")                 
                        plt.show()

                    # record diagnosis data
                    tf.logging.log_every_n(tf.logging.INFO,
                                           "[Global Step {}] [Epoch {}/{}: {}/{}] Minibatch Dice = {:.4f}, Minibatch MIoU = {:.4f}".format(
                                               global_step, epoch+1, epochs, iter+1, iters_per_epoch, loss, miou), 10)
                    self._record_summary(
                        summary_writer, 'training_loss', loss, global_step)
                    self._record_summary(
                        summary_writer, 'training_miou', miou, global_step)

                    # record global step
                    global_step = global_step + 1
                    iter += 1
                # output statistics for epoch
                cur_val_avg_miou = self._output_valstats(
                                    sess, summary_writer, epoch, valid_provider, valid_size, 
                                    "epoch_{}.mat".format(epoch+1), abs_prediction_path, ((epoch+1) % 200==0))

                if cur_val_avg_miou >= best:
                    best = cur_val_avg_miou
                    self.save(sess, save_path)

                self._record_summary(
                        summary_writer, 'best_miou', best, epoch+1)

                # save model
                if (epoch + 1) % save_epoch == 0:
                    directory = os.path.join(
                        abs_output_path, "{}_model/".format(epoch+1))
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    path = os.path.join(directory, "model")
                    self.save(sess, path)

            tf.logging.info('Training Ends')

    ###### Private Functions ######
    def dice_coef1(self, y_true, y_pred):
        dice_sum = 0
        smooth = self.data_kargs["measurement_smooth"]
        for i in range(self.data_kargs["oc"]):
            y_truef = y_true[:, :, :, i]
            y_predf = y_pred[:, :, :, i]
            And = tf.reduce_sum(y_truef * y_predf)
            dice_sum += ((2 * And + smooth) / (tf.reduce_sum(y_truef) + tf.reduce_sum(y_predf) + smooth)) \
                * self.loss_weights[i]
        return dice_sum # / np.sum(self.loss_weights[1:self.data_kargs["oc"]])

    def dice_coef(self, y_true, y_pred):
        smooth = self.data_kargs["measurement_smooth"]
        y_truef = K.flatten(y_true)
        y_predf = K.flatten(y_pred)
        And = tf.reduce_sum(y_truef * y_predf)
        # return (2 * And + smooth) \
        #     / (tf.reduce_sum(y_truef) + tf.reduce_sum(tf.abs(y_truef-y_predf)) + smooth)
        return (2 * And + smooth) \
            / (tf.reduce_sum(tf.square(y_truef)) + tf.reduce_sum(tf.square(y_predf)) + smooth)

    def dice_coef_change(self, y_true, y_pred):
        smooth = self.data_kargs["measurement_smooth"]
        y_truef = K.flatten(y_true)
        y_predf = K.flatten(y_pred)
        And = tf.reduce_sum(y_truef * y_predf)
        return (2 * And + smooth) \
            / (tf.reduce_sum(y_truef) + tf.reduce_sum(tf.abs(y_truef-y_predf)) + smooth)
        
    def dice_coef_loss(self, y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

    def dice_coef_loss_change(self, y_true, y_pred):
        return 1 - self.dice_coef_change(y_true, y_pred)
    
    def miou(self, y_true, y_pred):
        iou_sum = 0
        smooth = self.data_kargs["measurement_smooth"]
        for i in range(self.data_kargs["oc"]):
            intersection = tf.reduce_sum(y_true[:, :, :, i] * y_pred[:, :, :, i])
            # sum_ = tf.reduce_sum(y_true[:, :, :, i] + tf.abs(y_true[:, :, :, i] - y_pred[:, :, :, i]))
            sum_ = tf.reduce_sum(y_true[:, :, :, i] + y_pred[:, :, :, i])
            iou_sum += (intersection + smooth) / (sum_ - intersection + smooth) * self.loss_weights[i]
        return iou_sum / np.sum(self.loss_weights[:self.data_kargs["oc"]])
    
    def jac_distance(self, y_true, y_pred):
        return - self.miou(y_true, y_pred)

    def tversky(self, y_true, y_pred, alpha=0.7):
        tversky_sum = 0
        smooth = self.data_kargs["measurement_smooth"]
        for i in range(self.data_kargs["oc"]):
            y_truef = y_true[:, :, :, i]
            y_predf = y_pred[:, :, :, i]
            true_pos = tf.reduce_sum(y_truef * y_predf)
            false_neg = tf.reduce_sum(y_truef * (1 - y_predf))
            false_pos = tf.reduce_sum((1 - y_truef) * y_predf)
            tversky_sum += (true_pos + smooth) / \
                (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth) \
                * self.loss_weights[i]
            # tem = (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
            # low = tf.minimum(low, tem)
        return tversky_sum / np.sum(self.loss_weights)
        # return low              

    def tversky_loss(self, y_true, y_pred):
        return 1 - self.tversky(y_true, y_pred)
    
    def focal_tversky_loss(self, y_true, y_pred, alpha, gamma=0.75):
        tv = self.tversky(y_true, y_pred, alpha)
        return tf.pow((1 - tv), gamma)
    
    def _get_measure(self):

        # define the loss
        # loss = self.focal_tversky_loss(self.y, self.xhat,
        #                                self.data_kargs["loss_alpha"],
        #                                self.data_kargs["loss_gamma"])
        loss = self.dice_coef_loss(self.y, self.xhat)
        # loss = self.dice_coef_loss_change(self.y, self.xhat)
        # loss = tf.losses.sigmoid_cross_entropy(self.y, self.xhat, 0.1)
        # loss = self.cross_entropy(self.y, self.xhat)
        # loss = tf.keras.metrics.categorical_crossentropy(self.y, self.xhat)
        # loss = tf.nn.softmax_cross_entropy_with_logits(
        #     labels=self.y,
        #     logits=self.xhat,
        #     dim=-1,
        #     name="sce"
        # )
        # loss = K.mean(loss)
        # compute the miou
        miou = self.miou(self.y, self.xhat)
        return loss, miou

    def _output_valstats(self, 
                         sess, 
                         summary_writer, 
                         step, 
                         valid_provider,
                         batch_size,
                         name, 
                         save_path,
                         showPred):
        raw_iters = valid_provider.file_count / batch_size
        iters_per_epoch = int(raw_iters) + 1 if raw_iters > int(raw_iters) else int(raw_iters)

        loss = 0
        miou = 0
        for iter in range(iters_per_epoch):
            # select validation dataset (1 is dummy placeholder)
            valid_x, valid_y = valid_provider(batch_size, iter)
            
            loss_r, miou_r, img = sess.run([self.loss, self.miou, self.xhat],
                                           feed_dict={self.x: valid_x,
                                           self.y: valid_y})
            tf.logging.log_every_n(tf.logging.INFO,
                                   "validation_compute... {}/{}".format(iter, iters_per_epoch),
                                   20)
            if(showPred and ((iter+1) % 20 == 0)):
                for i in range(valid_x.shape[0]):
                    plt.figure(figsize=(15, 2))
                    plt.subplot(1, 3, 1)
                    plt.imshow(valid_x[i, :, :, 0], cmap="gray")
                    plt.subplot(1, 3, 2)
                    plt.imshow(valid_y[i, :, :, 0], cmap="gray")
                    plt.text(10, 10, f"{np.min(valid_y[i, :, :, 0]), np.max(valid_y[i, :, :, 0])}", color="green")
                    plt.subplot(1, 3, 3)
                    plt.imshow(img[i, :, :, 0], cmap="gray")
                    plt.text(10, 10, f"{np.min(img[i, :, :, 0]), np.max(img[i, :, :, 0])}", color="blue")
                plt.show()
            loss += loss_r
            miou += miou_r

        loss = loss/iters_per_epoch
        miou = miou/iters_per_epoch

        self._record_summary(
            summary_writer, 'validation_loss', loss, step)
        self._record_summary(
            summary_writer, 'validation_miou', miou, step)

        tf.logging.info(
            "Validation Statistics, Validation Loss= {:.4f}, Validation MIoU= {:.4f}".format(loss, miou))

        # # get recon
        # num_dete = self.data_kargs['num_dete']
        # num_proj = self.data_kargs['num_proj']
        # k = 0
        # sinogram = np.zeros([num_dete, num_proj])
        # for i in range(num_dete):
        #     for j in range(num_proj):
        #         sinogram[i,j] = xhat[k]
        #         k = k+1
        # theta = np.linspace(0., 180., num_proj, endpoint=False)
        # recon = iradon(sinogram, theta=theta, circle=True)
        # spio.savemat(os.path.join(save_path, name), {'recon': recon, 'sinogram': sinogram})
        return miou

    @staticmethod
    def _path_checker(output_path, 
                      prediction_path, 
                      is_restore):
        abs_prediction_path = os.path.abspath(prediction_path)
        abs_output_path = os.path.abspath(output_path)

        if not is_restore:
            tf.logging.info("Removing '{:}'".format(abs_prediction_path))
            shutil.rmtree(abs_prediction_path, ignore_errors=True)
            tf.logging.info("Removing '{:}'".format(abs_output_path))
            shutil.rmtree(abs_output_path, ignore_errors=True)

        if not os.path.exists(abs_prediction_path):
            tf.logging.info("Allocating '{:}'".format(abs_prediction_path))
            os.makedirs(abs_prediction_path)

        if not os.path.exists(abs_output_path):
            tf.logging.info("Allocating '{:}'".format(abs_output_path))
            os.makedirs(abs_output_path)

        return abs_output_path, abs_prediction_path

    @staticmethod
    def _record_summary(writer, 
                        name, 
                        value, 
                        step):

        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=value)
        writer.add_summary(summary, step)
        writer.flush()
