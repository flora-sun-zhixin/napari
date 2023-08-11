# Flora Sun, CIG, wustl, 2021
import tensorflow as tf

def computeConfusionMatrix(y_true, y_pred, containsBackground=False, precision=tf.float32, separateChannels=False):
    """
    Notice: the first channel of y_pred is expected to be the likelihood of being background
    """
    assert y_true.shape[:-1] == y_pred.shape[:-1], "shape mismatch of y_true:{} and y_pred:{}".format(y_true.shape, y_pred.shape)
    # if y_true is sparse, one hot encoding
    if y_pred.shape[-1] != y_true.shape[-1] and y_true.shape[-1] == 1:
        num_channel = y_pred.shape[-1]
        y_true = tf.concat([tf.where(y_true == i, 1., 0.) for i in range(num_channel)], axis=-1)
    # remove background channel if not contains background
    if not containsBackground:
        y_pred = y_pred[..., 1:]
        y_true = y_true[..., 1:]
    # cast data
    y_true = tf.cast(y_true, precision)
    y_pred = tf.cast(y_pred, precision)
    # specify the channel
    if separateChannels:
        axis = list(range(1, len(y_pred.shape) - 1))
    else:
        axis = list(range(1, len(y_pred.shape)))
    # compute the tp, tn, fp, fn
    tp = tf.reduce_sum(y_true * y_pred, axis=axis)
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=axis)
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=axis)
    tn = tf.reduce_sum((1 - y_true) * (1 - y_pred), axis=axis)

    return tp, fp, tn, fn

def dice(tp, fp, tn, fn, smooth=1e-5):
    return (2. * tp + smooth) / (2 * tp + fp + fn + smooth)

def jaccard(tp, fp, tn, fn, smooth=1e-5):
    """MIoU"""
    return (tp + smooth) / (tp + fp + fn + smooth)

def sensitivity(tp, fp, tn, fn, smooth=1e-5):
    """how many truely positive are predicted correctly"""
    return (tp + smooth) / (tp + fn + smooth)

def recall(tp, fp, tn, fn, smooth=1e-5):
    # same as sensitivity
    return sensitivity(tp, fp, tn, fn, smooth=smooth)

def specificity(tp, fp, tn, fn, smooth=1e-5):
    """how many truely negative are correctly predicted"""
    return (tn + smooth) / (tn + fp + smooth)

def positivePredictiveValue(tp, fp, tn, fn, smooth=1e-5):
    """Among all that is predicted as positive, how many among them are truely positive"""
    return (tp + smooth) / (tp + fp + smooth)

def precision(tp, fp, tn, fn, smooth=1e-5):
    # same as positivePredictiveValue
    return positivePredictiveValue(tp, fp, tn, fn, smooth=smooth)

def accuracy(tp, fp, tn, fn, smooth=1e-5):
    """Among all the pixels, how many are correctly predicted"""
    return (tp + tn + smooth) / (tp + tn + fp + fn + smooth)


ALL_METRICS = {
    "Dice": dice,
    "Jaccard": jaccard,
    "Sensitivity": sensitivity,
    "Recall": recall,
    "Specificity": specificity,
    "PositivePredictiveValue": positivePredictiveValue,
    "Precision": precision,
    "Accuracy": accuracy
}

default_metrics = ["Dice",
                   "Jaccard",
                   "Sensitivity",
                   "Recall",
                   "Specificity",
                   "PositivePredictiveValue",
                   "Precision",
                   "Accuracy"]


class MetricsList(tf.keras.metrics.Metric):
    def __init__(self,
                 metricsNameList=default_metrics,
                 mode="train",
                 separateChannels=False,
                 containsBackground=False,
                 precision=tf.float32,
                 smooth=1e-5,
                 **kwargs):
        super().__init__(**kwargs)
        self.metricsNameList = metricsNameList
        self.separateChannels = separateChannels
        self.containsBackground = containsBackground
        self.precision = precision
        self.smooth = smooth
        self.mode = mode

        self.totalRecords = [self.add_weight(mName + "_" + self.mode, initializer="zeros")
                             for mName in self.metricsNameList]
        self.count = self.add_weight("count_" + self.mode, initializer="zeros")

    def getMetricsNameList(self):
        return [(mName + "_" + self.mode) for mName in self.metricsNameList]

    def result(self):
        return [value / self.count for value in self.totalRecords]

    def update_state(self, y_true, y_pred):
        tp, fp, tn, fn = computeConfusionMatrix(y_true, y_pred,
                                                separateChannels=self.separateChannels,
                                                containsBackground=self.containsBackground,
                                                precision=self.precision)
        for i in range(len(self.totalRecords)):
            self.totalRecords[i].assign_add(
                tf.cast(tf.reduce_sum(
                    ALL_METRICS[self.metricsNameList[i]](tp, fp, tn, fn, smooth=self.smooth),
                    axis=0), tf.float32))
        self.count.assign_add(y_true.shape[0], tf.float32)

    def reset_states(self):
        for value in self.totalRecords:
            value.assign(0.)
        self.count.assign(0.)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "metricsNameList": self.metricsNameList,
                "mode": self.mode,
                "separateChannels": self.separateChannels,
                "containsBackground": self.containsBackground,
                "precision": self.precision,
                "smooth": self.smooth}

    
class MIoU(tf.keras.metrics.Metric):
    def __init__(self, smooth=10, class_weights=None, num_channel=None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is None:
            self.class_weights = tf.ones((num_channel, 1))
        else:
            class_weights = tf.cast(class_weights, tf.float32)
            if tf.reduce_prod(class_weights.shape) == num_channel:
                self.class_weights = tf.reshape(class_weights, (num_channel, 1))
            else:
                tf.print("The shape of the weights is not right, reduce to even weights")
                self.class_weights = tf.ones((num_channel, 1))
        self.smooth = smooth
        self.total_MIoU = self.add_weight("total_MIoU", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

        def MIoU(y_true, y_pred):
            And = tf.reduce_sum(y_true * y_pred, axis=[1,2])
            sum_true = tf.reduce_sum(tf.square(y_true), axis=[1,2])
            sum_pred = tf.reduce_sum(tf.square(y_pred), axis=[1,2])
            IoU_class = (And + self.smooth) / (sum_true + sum_pred - And + self.smooth)
            return IoU_class@self.class_weights / tf.reduce_sum(self.class_weights)

        self.MIoU = MIoU

    def update_state(self, y_true, y_pred):
        self.total_MIoU.assign_add(tf.reduce_sum(self.MIoU(y_true, y_pred)))
        self.count.assign_add(tf.cast(y_true.shape[0], tf.float32))

    def result(self):
        return self.total_MIoU / self.count

    def reset_states(self):
        self.total_MIoU.assign(0.)
        self.count.assign(0.)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "class_weights": self.class_weights, "smooth": self.smooth, "MIoU": self.MIoU}


class Sparse_MIoU(tf.keras.metrics.Metric):
    def __init__(self, smooth=10, **kwargs):
        super().__init__(**kwargs)
        self.smooth = tf.cast(smooth, tf.float64)
        self.total_MIoU = self.add_weight("total_MIoU", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

        def MIoU(y_true, y_pred):
            channels = y_pred.shape[-1]
            # y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred[..., 1:], tf.float64)
            y_true_mc = tf.concat([tf.where(y_true == i, 1., 0.) for i in range(1, channels)], axis = -1)
            y_true_mc = tf.cast(y_true_mc, tf.float64)
            MIoU = tf.constant([0.])
            # notice here that when we put the whole 3D matrix in, it would exceed the range of the float32,
            # so the computation of the background channel will be screwed up when computing all the channels in one go.
            And = tf.reduce_sum(y_pred * y_true_mc, axis=[1, 2, 3])
            sum_true = tf.reduce_sum(tf.square(y_true_mc), axis=[1, 2, 3])
            sum_pred = tf.reduce_sum(tf.square(y_pred), axis=[1, 2, 3])
            MIoU = tf.cast((And + self.smooth) / (sum_true + sum_pred - And + self.smooth), tf.float32)

            return tf.reduce_sum(MIoU / tf.constant(float(channels - 1)), axis=-1)

        self.MIoU = MIoU

    def update_state(self, y_true, y_pred):
        self.total_MIoU.assign_add(tf.reduce_sum(self.MIoU(y_true, y_pred)))
        self.count.assign_add(float(y_true.shape[0]))

    def result(self):
        return self.total_MIoU / self.count

    def reset_states(self):
        self.total_MIoU.assign(0.)
        self.count.assign(0.)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "class_weights": self.class_weights,
                "num_channel": self.num_channel,
                "smooth": self.smooth,
                "MIoU": self.MIoU}

