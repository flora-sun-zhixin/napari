# Flora Sun, WUSTL, 2021 
import os
import io
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as layers
from napari.components.CodeFromFlora.utils.utils import linear_decay, normalize, standardize
from napari.components.CodeFromFlora.models.AttentionGateLayer import MultiAttentionBlock

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_filter, num_conv=3, kernel_size=3, strides=1, use_resi=True,
                 batchNorm=False, bnkwargs=None, **kwargs):
        super().__init__(**kwargs)
        
        self.num_filter = num_filter
        self.num_conv = num_conv
        self.kernel_size = kernel_size
        self.strides = strides
        self.batchNorm = batchNorm
        self.use_resi = use_resi

        stridesList = [strides] + [1] * (num_conv - 1)
        self.convList = [layers.Conv2D(num_filter, kernel_size=kernel_size,
                                       strides=stridesList[i], padding="same") for i in range(num_conv)]

        self.activationLayer = [layers.LeakyReLU(alpha=0.01)] * num_conv

        self.bnList = None
        if self.batchNorm:
            self.bnList = [layers.BatchNormalization()] * num_conv

        self.conv_to_reshape = None
        if self.use_resi:
            self.conv_to_reshape = layers.Conv2D(num_filter, kernel_size=1,
                                                 strides=strides)

    def call(self, X, training=None):
        # Y = tf.keras.activations.relu(self.bn1(self.conv1(X), training=training))
        # Y = self.bn2(self.conv2(Y), training=training)
        Y = self.convList[0](X)

        for i in range(1, self.num_conv):
            if self.batchNorm:
                Y = self.bnList[i-1](Y, training=training)
            Y = self.activationLayer[i-1](Y)
            Y = self.convList[i](Y)

        if self.use_resi:
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
                "use_resi": self.resi}


class ResUNet(tf.keras.Model):
    def __init__(self,
                 ic=1,
                 oc=4,
                 padding=[[2, 2], [4, 5]],
                 depth=5,
                 start_exp=5,
                 num_conv=3):
        super().__init__()
        tf.keras.backend.clear_session()
        
        self.ic = ic
        self.oc = oc
        self.padding = padding
        self.depth = depth
        self.start_exp = start_exp
        self.num_conv = num_conv
        
        self.paddings = tf.constant([[0, 0], self.padding[0],
                                     self.padding[1], [0, 0]])
        # self.down_res = [Residual(2**i, num_conv=min(i - start_exp + 1, 3), name=f"down_res_{i - start_exp + 1}")
        #                  for i in range(start_exp, start_exp + depth)]
        self.down_res = [ConvBlock(2**i, num_conv=num_conv, use_resi=False, batchNorm=True, name=f"down_res_{i - start_exp + 1}")
                         for i in range(start_exp, start_exp + depth)]
        # self.up_res = [Residual(2**i, num_conv=min(i + 1 - start_exp, 3), name=f"up_res_{i - start_exp}")
        #                for i in range(start_exp - 2 + depth, start_exp - 1, -1)]
        self.up_res = [ConvBlock(2**i, num_conv, use_resi=False, batchNorm=True, name=f"up_res_{i - start_exp}")
                       for i in range(start_exp - 2 + depth, start_exp - 1, -1)]
        self.pool = [layers.MaxPool2D(pool_size=2, strides=2, padding="valid")
                     for _ in range(depth - 1)]
        self.bottomLinear = ConvBlock(2**(start_exp + depth - 1), num_conv=1, use_resi=False, batchNorm=True, kernel_size=1, name="bottomLinear")
        self.up_conv = [layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
                        for i in range(start_exp - 2 + depth, start_exp - 1, -1)]
        self.attentionBlock2D = [MultiAttentionBlock(in_size=2**i, gate_size=2**(i + 1), inter_size=2**i,
                                                     nonlocal_mode="concatenate", sub_sample_factor=(1, 1))
                                for i in range(start_exp - 2 + depth, start_exp - 1, -1)]
        self.outputLayer = layers.Conv2D(self.oc, kernel_size=1)
        self.loadedWeight = False
        self.attention = []

    def call(self, inputs, positionalEncoding, training=True):
        X = tf.pad(inputs, self.paddings, "CONSTANT")
        down_output = []
        for i in range(len(self.down_res) - 1):
            X = self.down_res[i](X)
            down_output.append(X)
            X = self.pool[i](X)

        X = self.down_res[-1](X)
        X = self.bottomLinear(X)

        self.attention = []
        for i in range(len(self.up_conv)):
            weightedShortCut, att = self.attentionBlock2D[i](down_output[self.depth - 2 - i], X)
            self.attention.append(att)
            X = self.up_conv[i](X)
            X = tf.concat([weightedShortCut, X], axis=-1)
            X = self.up_res[i](X)

        # X = tf.keras.activations.sigmoid(X)
        X = tf.concat([X, positionalEncoding], axis=-1)
        X = self.outputLayer(X)
        softmax = tf.keras.activations.softmax(X)

        # for situation where padding on the right is 0
        pad_end = []
        for i in range(2):
            pad_end.append(-self.padding[i][1]
                           if self.padding[i][1] != 0 else None)

        return softmax[:,
                       self.padding[0][0]: pad_end[0],
                       self.padding[1][0]: pad_end[1],
                       :]

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "ic": self.ic,
                "oc": self.oc,
                "padding": self.padding,
                "depth": self.depth,
                "start_exp": self.start_exp,
                "num_conv": self.num_conv}

    def loadWeight(self, best_dir):
        model = tf.train.get_checkpoint_state(best_dir)
        if model and model.model_checkpoint_path:
            tf.print("weights from ", best_dir, " have been loaded into the model")
            self.load_weights(model.model_checkpoint_path)
            self.loadedWeight = True

    def pred(self, best_dir, x_test, positionalEncoding, reloadWeights=False):
        if not self.loadedWeight or reloadWeights:
            self.loadWeight(best_dir)
        y_pred = self.call(x_test, positionalEncoding, training=False)
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
            start_epoch = start_epoch
            num_sample = trainDataset.reduce(0, lambda x,_: x+1).numpy()
            global_step = (num_sample // batch_size + int(num_sample % batch_size != 0)) * (start_epoch - 1)

        else:
            global_step = 1
            start_epoch = 1
        
        # tracking the model with the highest validation loss
        best = 0

        # optimizer
        optimizer = tf.keras.optimizers.Adam()
        val_mean_loss = tf.keras.metrics.Mean()

        # change of loss weights
        ceWeights = linear_decay(trainLossFunction.weight_ce, 0, epochs // 3) 

        # begin the training process
        tf.print("Start Training")
        padded_shapes = ([None],())
        for epoch in range(start_epoch, epochs + 1):
            if epoch <= (epochs // 3):
                trainLossFunction.setWeightCE(ceWeights[epoch])
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

            for step, batch in enumerate(augTrainDataset.shuffle(50).batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)):
                step += 1
                batch_x, batch_pos, batch_y, batch_cls = batch["image"], batch["positionalEncoding"], batch["mask"], batch["class"]
  
                with tf.GradientTape() as tape:
                    y_pred = self.call(batch_x, batch_pos, training=True)
                    loss, seg_dice, seg_ce = trainLossFunction(batch_y, y_pred, batch_cls)
                trainMetricsList(batch_y, y_pred)
                gradients = tape.gradient(loss, self.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                self._print_training_state_bar(epoch, epochs, step,
                                               trainLossFunction.name, loss, trainMetricsList)

                # write the training log
                self._log_summary_scale(log_writer, trainLossFunction.name, loss,
                                        seg_dice, seg_ce, "train", learning_rate[epoch-1], trainMetricsList, global_step)
                if global_step % 100 == 0:
                    self._log_summary_image(log_writer, batch_x, batch_y, y_pred, global_step, "predict_train")
                    self._log_summary_image(log_writer, batch_x, batch_y,
                                            tf.reduce_mean(self.getAttention()[-1], axis=-1, keepdims=True),
                                            global_step, "attention_train", enum=False)
                trainMetricsList.reset_states()
                global_step += 1
                
            log_writer.flush()
            # validation set
            for step, batch in enumerate(augValidDataset.shuffle(50).batch(valid_size, drop_remainder=True).prefetch(AUTOTUNE)):
                step += 1
                batch_x, batch_pos, batch_y, batch_cls = batch["image"], batch["positionalEncoding"], batch["mask"], batch["class"]
                y_pred = self.call(batch_x, batch_pos, training=False)
                loss_val, seg_dice, seg_ce = validLossFunction(batch_y, y_pred, batch_cls)
                validMetricsList(batch_y, y_pred)
                val_mean_loss(loss_val)
                self._print_val_state_bar(step, val_mean_loss, validMetricsList)
                if step == 1:
                    self._log_summary_image(log_writer, batch_x, batch_y, y_pred, global_step, "predict_valid")
                    self._log_summary_image(log_writer, batch_x, batch_y,
                                            tf.reduce_mean(self.getAttention()[-1], axis=-1, keepdims=True),
                                            global_step, "attention_valid", enum=False)

            self._log_summary_scale(log_writer, validLossFunction.name, loss_val,
                                    seg_dice, seg_ce, "valid", None, validMetricsList, global_step)

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

    def getAttention(self):
        return self.attention

    @staticmethod
    def randomSelect(batch_x, batch_y, y_pred, numGenerates):
        """
        randomly select numGenerates of slices along the given direction. return the corresponding image, gdtmask and pred.
        """
        batches = batch_x.shape[0]
        perm = tf.range(batches)
        if perm.shape[0] < numGenerates:
            perm = tf.repeat(perm, numGenerates)
        randomBatch = tf.random.shuffle(perm)[:numGenerates]

        return (tf.gather(batch_x[..., 0], randomBatch, axis=0).numpy(),
                tf.gather(batch_y[..., 0], randomBatch, axis=0).numpy(),
                tf.gather( y_pred[..., 0], randomBatch, axis=0).numpy())

    @staticmethod
    def image_grid(batch_x, batch_y, y_pred, enum):
        """
        random select some slices among x, y and pred and plot
        """
        numGenerates = 3
        sx, sygdt, sypred = ResUNet.randomSelect(batch_x, batch_y, y_pred, numGenerates)
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
            ax.imshow(sygdt[i, :, :], cmap="hot", interpolation="none", vmin=0, vmax=7)
            ax.axis("off")
            ax = axes[i][2]
            vmax = 7 if enum else None
            vmin = 0 if enum else None
            ax.imshow(sypred[i, :, :], cmap="hot", interpolation="none", vmin=vmin, vmax=vmax)
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
    def _log_summary_scale(writer, loss_name, loss, seg_dice, seg_ce, mode, learning_rate, metrics, step):
        with writer.as_default():
            metricsResult = metrics.result()
            for i, mName in enumerate(metrics.getMetricsNameList()):
                tf.summary.scalar(mName, metricsResult[i], step=step)
            if mode == "train":
                tf.summary.scalar("learning_rate", learning_rate, step=step)

            tf.summary.scalar(loss_name, loss, step=step)
            tf.summary.scalar("seg_Dice_" + mode, seg_dice, step=step)
            tf.summary.scalar("seg_CrossEntropy_" + mode, seg_ce, step=step)
            
    @staticmethod
    def _log_summary_image(writer, batch_x, batch_y, y_pred, step, name, enum=True):
        if batch_y.shape[-1] > 1:
            # enumerate the batch_y to display
            batch_y = tf.argmax(batch_y, axis=-1)[..., tf.newaxis]
        if enum:
            # enumerate the pred_y to display
            y_pred = tf.argmax(y_pred, axis=-1)[..., tf.newaxis]

        with writer.as_default():
            tf.summary.image(name,
                             ResUNet.plot_to_image(ResUNet.image_grid(batch_x, batch_y, y_pred, enum)),
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

