# Flora Sun, WUSTL, 2021
import tensorflow as tf
from napari.components.CodeFromFlora.utils.utils import putWeightOnBound

def init_class_weight(class_weights, num_channel, contains_background, precision, loss_name):
    num_channel = num_channel if contains_background else (num_channel - 1)
    if class_weights is None:
        class_weights = tf.ones((num_channel,), dtype=precision)
    else:
        class_weights = class_weights if contains_background else class_weights[1:]
        if tf.reduce_prod(class_weights.shape) == num_channel:
            class_weights = tf.reshape(class_weights, (num_channel,))
        else:
            tf.print(f"WARNING: in {loss_name} The shape of the weights {class_weights.shape} does not meet the num of channels {num_channel}, reduce to even weights")

    class_weights = tf.cast(class_weights, precision)
    return class_weights
                                      
class Dice_Loss_Base(tf.keras.losses.Loss):
    def __init__(self, smooth=1, precision=tf.float32, name="Dice_Loss_Base", **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = tf.cast(smooth, precision)
        self.precision = precision
        self.dice_denominator = 1.
        self.dice_numerator = 0.
        
    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, self.precision)
        y_true = tf.cast(y_true, self.precision)

        axis = list(range(1, y_pred.ndim - 1))

        tp = tf.reduce_sum(y_true * y_pred, axis=axis)
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=axis)
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=axis)

        self.dice_numerator = (2. * tp)
        self.dice_denominator = (2 * tp + fp + fn)

        return - tf.reduce_mean(self.dice_numerator / (self.dice_denominator + self.smooth))
    
    def get_dice_denominator(self):
        return self.dice_denominator

    def get_dice_numerator(self):
        return self.dice_numerator

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "smooth": self.smooth, "precision": self.precision}

class Customized_Dice_Loss(Dice_Loss_Base):
    def __init__(self, num_channel, contains_background=False, class_weights=None, use_bound_weight=False, bound_weight=1., precision=tf.float32, smooth=1, name="Weighted_Dice_Loss", **kwargs):
        super().__init__(smooth, precision, name=name, **kwargs)
        self.class_weights = init_class_weight(class_weights, num_channel, contains_background, precision, name)
        self.contains_background = contains_background
        self.use_bound_weight = use_bound_weight
        self.bound_weight = bound_weight

    def call(self, y_true, y_pred):
        # if y_true is enumerated, change to one hot
        if y_true.shape[-1] != y_pred.shape[-1] and y_true.shape[-1] == 1: # the case where y_true is enumerated
            # one hot encode
            y_true = tf.concat([tf.where(y_true == i, 1., 0.)
                                for i in range(y_pred.shape[-1])], axis=-1)
            print("turn y to one hot in dice loss")

        # whether remove background
        starting_channel = 0 if self.contains_background else 1
        y_true = y_true[..., starting_channel:]
        y_pred = y_pred[..., starting_channel:]

        # precision
        y_true = tf.cast(y_true, self.precision)
        y_pred = tf.cast(y_pred, self.precision)

        # whether use bound weights on the bound
        if self.use_bound_weight:
            boundWeight = tf.cast(putWeightOnBound(y_true, bound_weight), self.precision)
            y_true = y_true * boundWeight
            y_pred = y_pred * boundWeight

        # compute the loss
        dice = super().call(y_true, y_pred)
        return dice

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "class_weights": self.class_weights,
                "contains_background": self.contains_background,
                "use_bound_weight": self.use_bound_weight,
                "bound_weight": self.bound_weight}

class CrossEntropy_Loss_Base(tf.keras.losses.Loss):
    def __init__(self, precision=tf.float32, name="CrossEntropy_Base", **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = precision

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, self.precision)
        y_pred = tf.cast(y_pred, self.precision)
        epsilon_ = tf.constant(tf.keras.backend.epsilon(), dtype=y_pred.dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon_, 1 - epsilon_)
        self.separateChannel = tf.reduce_sum(y_true * tf.math.log(y_pred), axis=list(range(1, y_pred.ndim - 1)))
        return - tf.reduce_mean(self.separateChannel)

    def get_crossEntropy_separateChannel(self):
        return self.separateChannel

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "precision": precision}
    
class Customized_CrossEntropy_Loss(CrossEntropy_Loss_Base):
    def __init__(self, num_channel, contains_background=True, class_weights=None, use_bound_weight=False, bound_weight=1., precision=tf.float32, name="Weighted_CrossEntropy", **kwargs):
        super().__init__(precision=precision, name=name, **kwargs)
        self.class_weights = init_class_weight(class_weights, num_channel, contains_background, precision, name)
        self.contains_background = contains_background
        self.use_bound_weight = use_bound_weight
        self.bound_weight = bound_weight

    def call(self, y_true, y_pred):
        # the case where y_true is enumerated
        if y_true.shape[-1] != y_pred.shape[-1] and y_true.shape[-1] == 1: 
            y_true = tf.concat([tf.where(y_true == i, 1., 0.)
                               for i in range(y_pred.shape[-1])], axis=-1)

        # remove background if needed
        starting_channel = 0 if self.contains_background else 1
        y_true = y_true[..., starting_channel:]
        y_pred = y_pred[..., starting_channel:]

        # precision
        y_true = tf.cast(y_true, self.precision)
        y_pred = tf.cast(y_pred, self.precision)

        # add bound weight if needed
        if self.use_bound_weight:
            boundWeight = tf.cast(putWeightOnBound(y_true, self.bound_weight), self.precision)
            y_true = y_true * boundWeight

        ce = super().call(y_true, y_pred)
        return ce

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "class_weights": self.class_weights,
                "contains_background": self.contains_background,
                "use_bound_weight": self.use_bound_weight,
                "bound_weight": self.bound_weight}

class ClassificationCrossEntropyBinary(CrossEntropy_Loss_Base):
    def __init__(self, num_channel, contains_background=True, class_weights=None, precision=tf.float32, name="Classification_BCE", **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_weights = init_class_weight(class_weights, num_channel, contains_background,
                                               precision, name)
        self.precision = precision
        self.num_channel = num_channel
        
    def call(self, y_true, y_pred):
        ce = super().call(y_true, y_pred)
        return ce

    def __call__(self, y_true, y_pred):
        return self.call(y_true, y_pred)
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "class_weights": self.class_weights,
                "num_channel": self.num_channel}

class Filtered_DC_and_CE_loss(tf.keras.losses.Loss):
    def __init__(self, dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, smooth=1, precision=tf.float32, name="dc_and_ce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.smooth = smooth
        self.precision = precision
        if weight_ce != 0:
            self.ce = Customized_CrossEntropy_Loss(**ce_kwargs)
        else:
            self.ce = None
        if weight_dice != 0: 
            self.dice = Customized_Dice_Loss(**ce_kwargs)
        else:
            self.dice = None

    def call(self, y_true, y_pred, y_true_cls):
        y_true_cls = tf.cast(y_true_cls, self.precision)
        y_true_cls_ce = y_true_cls if ((self.ce is not None) and (self.ce.contains_background)) else y_true_cls[..., 1:]
        y_true_cls_dice = y_true_cls if ((self.dice is not None) and (self.dice.contains_background)) else y_true_cls[..., 1:]

        if self.weight_dice != 0:
            dice_loss = self.dice(y_true, y_pred)
            dice_num = self.dice.get_dice_numerator()
            dice_dem = self.dice.get_dice_denominator()
            dice_loss = tf.reduce_mean(
                tf.reduce_sum(y_true_cls_dice * dice_num * self.dice.class_weights, axis=-1) / \
                (tf.reduce_sum(y_true_cls_dice * dice_dem * self.dice.class_weights, axis=-1) + self.smooth), axis = 0)
        else:
            dice_loss = 0
        if self.weight_ce != 0:
            ce_loss = self.ce(y_true, y_pred)
            ce_separateChannel = self.ce.get_crossEntropy_separateChannel()
            ce_loss = tf.reduce_mean(
                tf.reduce_sum(y_true_cls_ce * ce_separateChannel * self.ce.class_weights, axis=-1) /\
                (tf.reduce_sum(y_true_cls_ce * self.ce.class_weights, axis=-1) + self.smooth), axis=0)
        else:
            ce_loss = 0
                        
        return  - tf.cast(ce_loss * self.weight_ce + dice_loss * self.weight_dice, tf.float32), \
            - tf.cast(tf.reduce_mean(dice_loss), tf.float32), \
            - tf.cast(tf.reduce_mean(ce_loss), tf.float32)

    def setWeightCE(self, weight_ce):
        self.weight_ce = weight_ce

    def __call__(self, y_true, y_pred, y_true_cls):
        return self.call(y_true, y_pred, y_true_cls)

class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self,
                 ClassCrossEntropy_kwargs,
                 DCandCELoss_kwargs,
                 weight_cls=1,
                 weight_seg=1,
                 name="combinedLoss",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.weight_cls = tf.cast(weight_cls, tf.float32)
        self.weight_seg = tf.cast(weight_seg, tf.float32)
        self.cls = ClassificationCrossEntropyBinary(**ClassCrossEntropy_kwargs)
        self.seg = DC_and_CE_loss(**DCandCELoss_kwargs)

    def call(self, y_true_cls, y_pred_cls, y_true, y_pred):
        if self.weight_cls == 0:
            cls_loss = 0
        else:
            cls_loss = self.cls(y_true_cls, y_pred_cls)
        if self.weight_seg == 0:
            seg_loss, seg_dice, seg_ce = 0, 0, 0
        else:
            seg_loss, seg_dice, seg_ce = self.seg(y_true, y_pred)
        y_true_cls = tf.cast(y_true_cls, tf.float32)
        epsilon_ = tf.constant(tf.keras.backend.epsilon(), dtype=y_pred.dtype)
        y_true_cls = tf.clip_by_value(y_true_cls, epsilon_, 1-epsilon_)

        y_true_cls_ce = y_true_cls if ((self.seg.ce is not None) and (self.seg.ce.contains_background)) else y_true_cls[..., 1:]
        y_true_cls_dice = y_true_cls if ((self.seg.dice is not None) and (self.seg.dice.contains_background)) else y_true_cls[..., 1:]
        diceMean = tf.reduce_mean(
            tf.reduce_sum(
                y_true_cls_dice * seg_dice * self.seg.weight_dice * self.seg.dice.class_weights / \
                tf.reduce_sum(y_true_cls_dice * self.seg.dice.class_weights), axis=-1)) if self.seg.dice is not None else 0
        ceMean = tf.reduce_mean(
            tf.reduce_sum(
                y_true_cls_ce * seg_ce * self.seg.weight_ce * self.seg.ce.class_weights / \
                tf.reduce_sum(y_true_cls_ce * self.seg.ce.class_weights), axis=-1)) if self.seg.ce is not None else 0
        clsMean = tf.reduce_mean(cls_loss)
        return  - self.weight_cls * clsMean + tf.reduce_max(y_true_cls[..., 1:]) * self.weight_seg * (ceMean + diceMean),\
            tf.reduce_mean(clsMean), \
            tf.reduce_mean(diceMean), \
            tf.reduce_mean(ceMean)

    def __call__(self, y_true_cls, y_pred_cls, y_true, y_pred):
        return self.call(y_true_cls, y_pred_cls, y_true, y_pred)

    def setWeightCE(self, weight_ce):
        self.seg.setWeightCE(tf.cast(weight_ce, tf.float32))
