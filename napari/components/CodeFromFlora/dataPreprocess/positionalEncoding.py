import math
import tensorflow as tf
import numpy as np

class PositionEmbeddingSine3D():
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    Code come from: https://github.com/facebookresearch/detr/blob/eb9f7e03ed8e2ed2cd55528989fe7df890bc3fc0/models/position_encoding.py
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def generatePosEncoding(self, dataInstanceDic: dict):
        x = dataInstanceDic["image"] # image shape: (axis0, axis1, axis2, channel)
        tem = np.ones(x.shape[:3])
        x_embed = np.cumsum(tem, 0, dtype=np.float32)
        y_embed = np.cumsum(tem, 1, dtype=np.float32)
        z_embed = np.cumsum(tem, 2, dtype=np.float32)
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[-1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            z_embed = z_embed / (z_embed[:, :, -1:] + eps) * self.scale

        dim_t = np.arange(self.num_pos_feats, dtype=np.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, None] / dim_t
        pos_x = np.stack((np.sin(pos_x[:, :, :, 0::2]), np.cos(pos_x[:, :, :, 1::2])), axis=-1)
        pos_x = pos_x.reshape(x.shape[0], x.shape[1], x.shape[2], -1)
        pos_y = np.stack((np.sin(pos_y[:, :, :, 0::2]), np.cos(pos_y[:, :, :, 1::2])), axis=-1)
        pos_y = pos_y.reshape(x.shape[0], x.shape[1], x.shape[2], -1)
        pos_z = np.stack((np.sin(pos_z[:, :, :, 0::2]), np.cos(pos_z[:, :, :, 1::2])), axis=-1)
        pos_z = pos_z.reshape(x.shape[0], x.shape[1], x.shape[2], -1)
        pos = np.concatenate((pos_y, pos_x, pos_z), axis=3)
        return pos

class PositionEmbeddingSine2DwithSliceNum():
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    Code come from: https://github.com/facebookresearch/detr/blob/eb9f7e03ed8e2ed2cd55528989fe7df890bc3fc0/models/position_encoding.py
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def generatePosEncoding(self, dataInstanceDic: dict):
        x = dataInstanceDic["image"] # image shape: (axis0, axis1, channel)
        axis0, axis1, _ = dataInstanceDic["imageShape"]
        tem = tf.ones((1, axis0, axis1), dtype=tf.float32)
        x_embed = tf.cumsum(tem, 0) * (dataInstanceDic["sliceNum"] + 1)
        y_embed = tf.cumsum(tem, 1)
        z_embed = tf.cumsum(tem, 2)
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (tf.ones((1, axis0, axis1), dtype=tf.float32) * dataInstanceDic["totalSliceNum"] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            z_embed = z_embed / (z_embed[:, :, -1:] + eps) * self.scale

        dim_t_x = tf.range(self.num_pos_feats, dtype=np.float32)
        dim_t_x = self.temperature ** (2 * (dim_t_x // 2) / self.num_pos_feats)
        dim_t_yz = dim_t_x[:dim_t_x.shape[0] // 2] * 2

        pos_x = x_embed[:, :, :, None] / dim_t_x
        pos_y = y_embed[:, :, :, None] / dim_t_yz
        pos_z = z_embed[:, :, :, None] / dim_t_yz
        pos_x = tf.stack((tf.math.sin(pos_x[:, :, :, 0::2]), tf.math.cos(pos_x[:, :, :, 1::2])), axis=-1)
        pos_x = tf.reshape(pos_x, (axis0, axis1, -1))
        pos_y = tf.stack((tf.math.sin(pos_y[:, :, :, 0::2]), tf.math.cos(pos_y[:, :, :, 1::2])), axis=-1)
        pos_y = tf.reshape(pos_y, (axis0, axis1, -1))
        pos_z = tf.stack((tf.math.sin(pos_z[:, :, :, 0::2]), tf.math.cos(pos_z[:, :, :, 1::2])), axis=-1)
        pos_z = tf.reshape(pos_z, (axis0, axis1, -1))
        pos = tf.concat((pos_y, pos_x, pos_z), axis=-1)
        return pos

class PositionEmbeddingSine1DwithSliceNum():
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    Code come from: https://github.com/facebookresearch/detr/blob/eb9f7e03ed8e2ed2cd55528989fe7df890bc3fc0/models/position_encoding.py
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def generatePosEncoding(self, dataInstanceDic: dict):
        x = dataInstanceDic["image"] # image shape: (axis0, axis1, channel)
        print(dataInstanceDic["imageShape"])
        axis0, axis1, _ = dataInstanceDic["imageShape"]
        #print(axis0.eval(session=tf.compat.v1.Session()))
        tem = tf.ones((1, axis0, axis1), dtype=tf.float32)
        x_embed = tf.cumsum(tem, 0) * (dataInstanceDic["sliceNum"] + 1)
        if self.normalize:
            x_embed = x_embed / (tf.ones((1, axis0, axis1), dtype=tf.float32) * dataInstanceDic["totalSliceNum"]) * self.scale

        dim_t_x = tf.range(self.num_pos_feats, dtype=np.float32)
        dim_t_x = self.temperature ** (2 * (dim_t_x // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t_x
        pos_x = tf.stack((tf.math.sin(pos_x[:, :, :, 0::2]), tf.math.cos(pos_x[:, :, :, 1::2])), axis=-1)
        pos_x = tf.reshape(pos_x, (axis0, axis1, -1))
        return pos_x
