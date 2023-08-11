# Flora Sun, CIG 2022 September,
# adapted from https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/layers/grid_attention_layer.py

import os
import shutil
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

class gridAttentionBlockND(tf.keras.layers.Layer):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3, mode="concatenate", \
                 sub_sample_factor=(2,2,2)):
        super(gridAttentionBlockND, self).__init__()
        assert dimension in [2, 3]
        assert mode in ["concatenate"]

        if isinstance(sub_sample_factor, tuple): self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list): self.sub_sample_factor = tuple(sub_sample_factor)
        else: self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        # if not set the kernel size to be the at least the sub_sample_factor, we will lose info because of stride
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = layers.Conv3D
            bn = layers.BatchNormalization
            self.upsample = layers.UpSampling3D
            self.upsampleMethod = "trilinear"
        elif dimension == 2:
            conv_nd = layers.Conv2D
            bn = layers.BatchNormalization
            self.upsample = layers.UpSampling2D
            self.upsampleMethod = "bilinear"
        else:
            raise NotImplemented

        initializer = tf.keras.initializers.HeNormal()

        # Output transform
        self.W = tf.keras.Sequential(
            [conv_nd(self.in_channels, kernel_size=1, strides=1, padding="same", kernel_initializer=initializer),
             bn()]
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(self.inter_channels, kernel_size=self.sub_sample_kernel_size,
                             strides=self.sub_sample_factor, padding="same", use_bias=False, kernel_initializer=initializer)
        self.phi = conv_nd(self.inter_channels, kernel_size=1, strides=1, padding="same", use_bias=True, kernel_initializer=initializer)
        self.psi = conv_nd(1, kernel_size=1, strides=1, padding="same", use_bias=True, kernel_initializer=initializer)

        # Define the operation
        if mode == 'concatenate':
            self.operation_function = self._concatenation
        else:
            raise NotImplementedError('Unknown operation function.')

    def call(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''

        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.shape
        batch_size = input_size[0]
        assert batch_size == g.shape[0]

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.shape

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = self.upsample(size=2, interpolation=self.upsampleMethod)(self.phi(g))
        f = tf.keras.activations.relu(theta_x + phi_g)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = tf.keras.activations.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        # sigm_psi_f = self.upsample(size=2, interpolation=self.upsampleMethod)(sigm_psi_f)
        y = tf.broadcast_to(sigm_psi_f, input_size) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


class GridAttentionBlock2D(gridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenate',
                 sub_sample_factor=(2,2)):
        super(GridAttentionBlock2D, self).__init__(in_channels,
                                                   inter_channels=inter_channels,
                                                   gating_channels=gating_channels,
                                                   dimension=2, mode=mode,
                                                   sub_sample_factor=sub_sample_factor,
                                                   )


class GridAttentionBlock3D(gridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenate',
                 sub_sample_factor=(2,2,2)):
        super(GridAttentionBlock3D, self).__init__(in_channels,
                                                   inter_channels=inter_channels,
                                                   gating_channels=gating_channels,
                                                   dimension=3, mode=mode,
                                                   sub_sample_factor=sub_sample_factor,
                                                   )


class MultiAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlock2D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor= sub_sample_factor)
        self.gate_block_2 = GridAttentionBlock2D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor=sub_sample_factor)
        self.combine_gates = tf.keras.Sequential([layers.Conv2D(in_size, kernel_size=1, strides=1, padding="same",
                                                                kernel_initializer=tf.keras.initializers.HeNormal()),
                                                  layers.BatchNormalization(),
                                                  layers.ReLU()])

    def call(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)
        gate_2, attention_2 = self.gate_block_2(input, gating_signal)

        return self.combine_gates(tf.concat([gate_1, gate_2], -1)), tf.concat([attention_1, attention_2], -1)
