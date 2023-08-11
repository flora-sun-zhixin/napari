# Flora Sun, WUSTL, 2021 
import os
import io
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as layers

from utils.utils import linear_decay
from utils.dataPreprocess import putBackOverlap, separateOverlap
# from line_profiler import LineProfiler

class Residual(tf.keras.layers.Layer):
    def __init__(self, num_filter, num_conv=3, kernel_size=3, strides=1,
                 use_1x1conv=True, batchNorm=False, bnkwargs=None, dropout=False, drpkwargs=None, **kwargs):
        super().__init__(**kwargs)
        
        self.num_filter = num_filter
        self.num_conv = num_conv
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_1x1conv = use_1x1conv
        self.batchNorm = batchNorm
        self.dropout = dropout

        if batchNorm and bnkwargs is None:
            bnkwargs = {"momentum":0.99, "epsilon":0.001, "center":True}

        if dropout and drpkwargs is None:
            drpkwargs = {"rate":0.25}

        stridesList = [strides] + [1] * (num_conv - 1)
        self.convList = [layers.Conv2D(num_filter, kernel_size=kernel_size,
                                       strides=stridesList[i], padding="same") for i in range(num_conv)]

        self.activationLayer = [layers.LeakyReLU(alpha=0.01)] * num_conv

        self.dropoutList = None
        if self.dropout:
            self.dropoutList = [layers.Dropout(**drpkwargs)] * num_conv

        self.bnList = None
        if self.batchNorm:
            self.bnList = [layers.BatchNormalization(**bnkwargs)] * num_conv

        self.conv_to_reshape = None
        if use_1x1conv:
            self.conv_to_reshape = layers.Conv2D(num_filter, kernel_size=1,
                                                 strides=strides)

    def call(self, X, training=None):
        # Y = tf.keras.activations.relu(self.bn1(self.conv1(X), training=training))
        # Y = self.bn2(self.conv2(Y), training=training)
        Y = self.convList[0](X)
        if self.dropout:
            Y = self.dropoutList[0](Y, training=training)

        for i in range(1, self.num_conv):
            Y = self.activationLayer[i-1](Y)
            if self.batchNorm:
                Y = self.bnList[i-1](Y, training=training)
            Y = self.convList[i](Y)
            if self.dropout:
                Y = self.dropoutList[i](Y, training=training)

        if self.conv_to_reshape is not None:
            X = self.conv_to_reshape(X)

        Y += X

        if self.batchNorm:
            Y = self.bnList[-1](Y, training=training)

        return self.activationLayer[-1](Y)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "num_filter": self.num_filter,
                "num_conv": self.num_conv,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "use_1x1conv": self.use_1x1conv}


class ResUNet(tf.keras.Model):
    def __init__(self,
                 data_kargs={'ic': 1,
                             'oc': 4,
                             'padding': [[2, 2], [4, 5]]},
                 net_kargs={'depth': 5,
                            'start_exp': 5,
                            'num_conv': 3}):
        super().__init__()
        tf.keras.backend.clear_session()
        
        self.data_kargs = data_kargs
        self.net_kargs = net_kargs
        
        s_exp = self.net_kargs["start_exp"]
        depth = self.net_kargs["depth"]
        self.paddings = tf.constant([[0, 0], self.data_kargs["padding"][0],
                                     self.data_kargs["padding"][1], [0, 0]])
        # self.down_res = [Residual(2**i, num_conv=min(i - s_exp + 1, 3), name=f"down_res_{i - s_exp + 1}")
        #                  for i in range(s_exp, s_exp + depth)]
        self.down_res = [Residual(2**i, num_conv=net_kargs["num_conv"], name=f"down_res_{i - s_exp + 1}")
                         for i in range(s_exp, s_exp + depth)]
        # self.up_res = [Residual(2**i, num_conv=min(i + 1 - s_exp, 3), name=f"up_res_{i - s_exp}")
        #                for i in range(s_exp - 2 + depth, s_exp - 1, -1)]
        self.up_res = [Residual(2**i, num_conv=net_kargs["num_conv"], name=f"up_res_{i - s_exp}")
                       for i in range(s_exp - 2 + depth, s_exp - 1, -1)]
        self.pool = [layers.MaxPool2D(pool_size=2, strides=2, padding="valid")
                     for _ in range(depth - 1)]
        self.up_conv = [layers.Conv2DTranspose(2**i, kernel_size=2, strides=2, padding="same")
                        for i in range(s_exp - 2 + depth, s_exp - 1, -1)]
        self.outputLayer1 = layers.Conv2D(self.data_kargs["oc"] + 1, kernel_size=1)
        # le-net part
        self.denseHidden1 = layers.Dense(512, activation="relu")
        self.denseHidden2 = layers.Dense(128, activation="relu")
        self.outputLayer2 = layers.Dense(data_kargs["oc"], activation="sigmoid")

        self.loadedWeight = False

    def call(self, inputs, training=True):
        img = inputs
        
        X = tf.pad(img, self.paddings, "CONSTANT")
        
        down_output = []
        for i in range(len(self.down_res) - 1):
            X = self.down_res[i](X)
            down_output.append(X)
            X = self.pool[i](X)

        X = self.down_res[-1](X)
        Y = layers.Flatten()(X)
        Y = self.denseHidden1(Y)
        Y = self.denseHidden2(Y)
        cls = self.outputLayer2(Y)
        
        for i in range(len(self.up_conv)):
            X = self.up_conv[i](X)
            X = tf.concat([down_output[self.net_kargs["depth"] - 2 - i], X],
                          axis=3)
            X = self.up_res[i](X)
                
        X = self.outputLayer1(X)
        softmax = tf.keras.activations.softmax(X)

        # put the overlap part back to the pelvis and the femur
        softmax = putBackOverlap(softmax, 5, 1, 4)
        
        # for situation where padding on the right is 0
        pad_end = []
        for i in range(2):
            pad_end.append(-self.data_kargs["padding"][i][1]
                           if self.data_kargs["padding"][i][1] != 0 else None)

        return cls, softmax[:,
                            self.data_kargs["padding"][0][0]: pad_end[0],
                            self.data_kargs["padding"][1][0]: pad_end[1],
                            :]

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "data_kargs": self.data_kargs, "net_kargs": self.net_kargs}

    def loadWeight(self, best_dir):
        model = tf.train.get_checkpoint_state(best_dir)
        if model and model.model_checkpoint_path:
            tf.print("weights from ", best_dir, " have been loaded into the model")
            self.load_weights(model.model_checkpoint_path)
            self.loadedWeight = True

    def pred(self, best_dir, x_test, reloadWeights=False):
        if not self.loadedWeight or reloadWeights:
            self.loadWeight(best_dir)
        y_pred = self.call(x_test, training=False)
        return y_pred

    def train(self,
              output_path,
              trainDataset,
              validDataset,
              augmentationFunciton,
              trainLossFunction,
              trainMetricsList,
              validLossFunction,
              validMetricsList,
              learning_rate=0.001,
              batch_size=20,
              valid_size=20,
              epochs=300,
              is_restore=False,
              load_path=None,
              save_epoch=10,
              start_epoch=1):
        if epochs == 0:
            tf.print("Parameter [epoch] is zero. Program terminated.")
            quit()

        abs_output_path = os.path.abspath(output_path)
        # directory for the best weights
        best_dir = os.path.join(abs_output_path, "best")
        if not os.path.exists(best_dir):
            os.makedirs(best_dir)
        best_weights = os.path.join(best_dir, "best_weights")
        # directory for the regular checkpoints save
        ckp_dir = os.path.join(abs_output_path, "ckp")
        if not os.path.exists(ckp_dir):
            os.makedirs(ckp_dir)
        ckp_weights = os.path.join(ckp_dir, "ckp_weights")
        # writer for the tensorboard writer
        log_writer = tf.summary.create_file_writer(self._get_log_dir(abs_output_path))

        # if restore -- load the best model
        if is_restore and load_path and os.path.exists(load_path):
            self.loadWeight(load_path)

        # tracking the model with the highest validation loss
        best = 0

        # optimizer
        optimizer = tf.keras.optimizers.Adam()
        val_mean_loss = tf.keras.metrics.Mean()

        # begin the training process
        tf.print("Start Training")
        global_step = 1

        # change of loss weights
        # ceWeights = linear_decay(trainLossFunction.seg.weight_ce, 0, epochs)
        
        for epoch in range(start_epoch, epochs + 1):
            # trainLossFunction.setWeightCE(ceWeights[epoch - 1])
            # training set
            for metric in (trainMetricsList, validMetricsList, val_mean_loss):
                metric.reset_states()
                
            # Learning rate
            if type(learning_rate) is np.ndarray:
                optimizer.learning_rate.assign(learning_rate[epoch - 1])
            elif type(learning_rate) is float:
                optimizer.learning_rate.assign(learning_rate)
            else:
                tf.logging.info(
                    'Learning rate should be a list of double or a double scalar.')
                quit()

            # data augmentation
            AUTOTUNE = tf.data.experimental.AUTOTUNE
            augTrainDataset = trainDataset.map(lambda x: augmentationFunciton(x, aug=tf.constant(True)), num_parallel_calls=AUTOTUNE) 
            augValidDataset = validDataset.map(lambda x: augmentationFunciton(x, aug=tf.constant(False)), num_parallel_calls=AUTOTUNE)

            for step, batch in enumerate(augTrainDataset.shuffle(50).batch(batch_size).prefetch(AUTOTUNE)):
                step += 1
                batch_x, batch_y, batch_cls = batch["image"], batch["mask"], batch["class"]
                with tf.GradientTape() as tape:
                    cls_pred, y_pred = self.call(batch_x, training=True)
                    loss, cls_loss, seg_dice, seg_ce = trainLossFunction(batch_cls, cls_pred, batch_y, y_pred)
                y_pred = y_pred * tf.round(cls_pred[:, tf.newaxis, tf.newaxis, :])
                trainMetricsList(batch_y, y_pred)
                gradients = tape.gradient(loss, self.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                self._print_training_state_bar(epoch, epochs, step,
                                               trainLossFunction.name, loss, trainMetricsList)
    
                # write the training log
                self._log_summary_scale(log_writer, trainLossFunction.name, loss, cls_loss,
                                        seg_dice, seg_ce, "train", trainMetricsList, global_step)
                if global_step % 100 == 0:
                    self._log_summary_image(log_writer, batch_x, batch_y, y_pred, cls_pred, global_step, mode="train")
                global_step += 1

            # validation set
            for step, batch in enumerate(augValidDataset.shuffle(50).batch(valid_size).prefetch(AUTOTUNE)):
                step += 1
                batch_x, batch_y, batch_cls = batch["image"], batch["mask"], batch["class"]
                cls_pred, y_pred = self.call(batch_x, training=False)
                loss_val, cls_loss, seg_dice, seg_ce = validLossFunction(batch_cls, cls_pred, batch_y, y_pred)
                y_pred = y_pred * tf.round(cls_pred[:, tf.newaxis, tf.newaxis, :])
                validMetricsList(batch_y, y_pred)
                val_mean_loss(loss_val)
                self._print_val_state_bar(step, val_mean_loss, validMetricsList)
                if step == 1:
                    self._log_summary_image(log_writer, batch_x, batch_y, y_pred, cls_pred, global_step, mode="valid")
            self._log_summary_scale(log_writer, validLossFunction.name, loss_val, cls_loss,
                                    seg_dice, seg_ce, "valid", validMetricsList, global_step)


            # save best model
            validDice = validMetricsList.result()[validMetricsList.getMetricsNameList().index("Dice_valid")]
            if validDice > best:
                self.save_weights(best_weights)
                best = validDice
                with log_writer.as_default():
                    tf.summary.scalar("best_loss", validDice, step=global_step)

            # save checkpoints
            if epoch % save_epoch == 0:
                self.save_weights(ckp_weights)

            log_writer.flush()

        tf.print("Training Ends")

    #  @profile
    def trail(self,
              output_path,
              trainDataset,
              validDataset,
              augmentationFunciton,
              trainLossFunction,
              trainMetricsList,
              validLossFunction,
              validMetricsList,
              learning_rate=0.001,
              batch_size=20,
              valid_size=20,
              epochs=300,
              is_restore=False,
              load_path=None,
              save_epoch=10,
              start_epoch=1):
        lp = LineProfiler()
        lp_wrapper = lp(self.train)
        lp_wrapper(output_path,
                   trainDataset,
                   validDataset,
                   augmentationFunciton,
                   trainLossFunction,
                   trainMetricsList,
                   validLossFunction,
                   validMetricsList,
                   learning_rate,
                   batch_size,
                   valid_size,
                   2,
                   is_restore,
                   load_path,
                   save_epoch,
                   start_epoch)
        lp.print_stats()
        
    @staticmethod
    def randomSelect(batch_x, batch_y, y_pred, cls_pred, numGenerates):
        """
        randomly select numGenerates of slices along the given direction. return the corresponding image, gdtmask and pred.
        """
        batches = batch_x.shape[0]
        perm = tf.range(batches)
        randomBatch = tf.random.shuffle(perm)[:numGenerates]
                
        if batch_y.shape[-1] == 1:
            pass
        else:
            batch_y = batch_y * tf.constant(range(1, batch_y.shape[-1]+1), dtype=batch_y.dtype)
            batch_y = tf.argmax(batch_y, axis=-1)[..., tf.newaxis]
        y_pred = tf.argmax(y_pred, axis=-1)[..., tf.newaxis]

        return (tf.gather(batch_x[..., 0], randomBatch, axis=0).numpy(),
                tf.gather(batch_y[..., 0], randomBatch, axis=0).numpy(),
                tf.gather( y_pred[..., 0], randomBatch, axis=0).numpy(),
                tf.gather(cls_pred, randomBatch, axis=0).numpy())

    @staticmethod
    def image_grid(batch_x, batch_y, y_pred, cls_pred):
        """
        random select some slices among x, y and pred and plot
        """
        numGenerates = 3
        y_pred = separateOverlap(y_pred, 1, 4)
        sx, sygdt, sypred, sclspred = ResUNet.randomSelect(batch_x, batch_y, y_pred, cls_pred, numGenerates)
        if sx.shape[0] < numGenerates:
            fig = plt.figure(figsize=(15, 15))
            return fig
        fig, axes = plt.subplots(nrows=numGenerates, ncols=3, figsize=(15, 15))
        cols = ["ctData", "true label", "pred label"]
        plt.subplots_adjust(wspace=5, hspace=5)
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)
        for i in range(numGenerates):
            ax = axes[i][0]
            ax.imshow(sx[i, :, :], cmap="gray", interpolation="none")
            ax.axis("off")
            ax = axes[i][1]
            ax.imshow(sygdt[i, :, :], cmap="hot", interpolation="none", vmin=0, vmax=y_pred.shape[-1])
            ax.axis("off")
            ax = axes[i][2]
            ax.imshow(sypred[i, :, :], cmap="hot", interpolation="none", vmin=0, vmax=y_pred.shape[-1])
            pred = list(map(lambda n: "%.2f"%n, list(sclspred[i])))
            ax.set_title(f"y: {tf.unique(tf.reshape(tf.round(sypred[i, :, :]), [-1]))[0]}, {pred}")
            ax.axis("off")
            ax.grid(False)

        fig.tight_layout()
        return fig

    @staticmethod
    def plot_to_image(figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image
            
    @staticmethod
    def _log_summary_scale(writer, loss_name, loss, cls_loss, seg_dice, seg_ce, mode, metrics, step):
        with writer.as_default():
            metricsResult = metrics.result()
            for i, mName in enumerate(metrics.getMetricsNameList()):
                tf.summary.scalar(mName, metricsResult[i], step=step)
            tf.summary.scalar(loss_name, loss, step=step)
            tf.summary.scalar("cls_CrossEntropy_" + mode, cls_loss, step=step)
            tf.summary.scalar("seg_Dice_" + mode, seg_dice, step=step)
            tf.summary.scalar("seg_CrossEntropy_" + mode, seg_ce, step=step)
            
    @staticmethod
    def _log_summary_image(writer, batch_x, batch_y, y_pred, cls_pred, step, mode="train"):
        with writer.as_default():
            tf.summary.image(f"predict_{mode}",
                             ResUNet.plot_to_image(ResUNet.image_grid(batch_x, batch_y, y_pred, cls_pred)),
                             step=step)

    @staticmethod
    def _get_log_dir(abs_output_path):
        import time
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(abs_output_path, run_id)

    @staticmethod
    def _print_training_state_bar(epoch, epochs, it, loss_name, loss, metrics):
        # print udpate on multiple lines
        num_lines = len(metrics.getMetricsNameList()) + 2
        # set up blank lines so cursor moves work
        if it == 1:
            print("\n" * num_lines + "\n")
        UP = f"\x1B[{num_lines}A" # move the curser up
        CLR = "\x1B[0K"           # erase remaining characters in the line 
        metricsResult = metrics.result()
        metrics_str = f",{CLR}\n".join("{}: {:.4f}".format(mName, metricsResult[i])
                                for i, mName in enumerate(metrics.getMetricsNameList()))
        metrics_str = "{}: {:.4f},{}\n".format(loss_name, loss, CLR) + metrics_str
        print("{}[Epoch {}/{}: {}]{}\n".format(
            UP, epoch, epochs, it, CLR) + metrics_str, end="\n")

    @staticmethod
    def _print_val_state_bar(val_it, loss, metrics):
        # print udpate on multiple lines
        num_lines = len(metrics.getMetricsNameList()) + 2
        # set up blank lines so cursor moves work
        if val_it == 1:
            print("\n" * num_lines + "\n")
        UP = f"\x1B[{num_lines}A" # move the curser up
        CLR = "\x1B[0K"           # erase remaining characters in the line 
        metricsResult = metrics.result()
        metrics_str = f",{CLR}\n".join("average {}: {:.4f} {}".format(mName, metricsResult[i], CLR)
                                for i, mName in enumerate(metrics.getMetricsNameList()))
        metrics_str = "{}: {:.4f},{}\n".format(loss.name, loss.result(), CLR) + metrics_str
        print("{}[Validation {}]{} \n".format(UP, val_it, CLR) + metrics_str, end="\n")
                
    @staticmethod
    def _show_prediction(y_true, y_pred):
        num_channel = y_true.shape[-1]
        plt.figure(figsize=(num_channel * 4, 2))
        for i in range(num_channel):
            plt.subplot(2, num_channel, i + 1)
            plt.imshow(y_true[0, :, :, i], cmap="gray", vmin=0, vmax=1)
            plt.axis("off")
            plt.subplot(2, num_channel, i + 1 + num_channel)
            plt.imshow(y_pred[0, :, :, i], cmap="gray", vmin=0, vmax=1)
            plt.axis("off")
        plt.show()

    @staticmethod
    def _path_checker(output_path, is_restore):
        abs_output_path = os.path.abspath(output_path)

        if not is_restore:
            tf.print("Removing '{:}'".format(abs_output_path))
            shutil.rmtree(abs_output_path, ignore_errors=True)

        if not os.path.exists(abs_output_path):
            tf.print("Allocating '{:}'".format(abs_output_path))
            os.makedirs(abs_output_path)

        return abs_output_path
