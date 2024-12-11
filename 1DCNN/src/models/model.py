# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:30:24 2022

@author: ras
"""


from audioop import add
from calendar import c
from cmath import cos
import os
from re import L, T
import re
import sys
import math
from turtle import st

import numpy as np
import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Reshape, Conv3D, Conv2D, Conv1D, Conv2DTranspose, Conv1DTranspose, BatchNormalization, Activation, GlobalAveragePooling1D, GlobalAveragePooling2D, AveragePooling2D, AveragePooling1D, MaxPooling1D, Lambda, Input, Concatenate, Add, UpSampling2D, UpSampling1D, LeakyReLU, ZeroPadding2D,Multiply, DepthwiseConv2D, MaxPooling2D, LayerNormalization
from tensorflow.keras.models import Model

# groupnormalization をtfaからimport



# from .efficientnetv2 import effnetv2_model

sys.path.append('../')
WEIGHT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"weights/")
USE_TPU = False

if USE_TPU:
    batch_norm = tf.keras.layers.experimental.SyncBatchNormalization
else:
    batch_norm = BatchNormalization
    # batch_norm = tfa.layers.GroupNormalization


def cbr_1d(x, out_layer, kernel, stride, name, bias=False, use_batchnorm=True):
    x = Conv1D(out_layer, kernel_size=kernel, strides=stride, use_bias=bias, padding="same", name=name+"_conv")(x)
    if use_batchnorm:
        x = batch_norm(name = name+"_bw")(x)
    x = Activation("relu", name=name+"_activation")(x)
    return x

def se_1d(x_in, layer_n, rate, name):
    x = GlobalAveragePooling1D(name=name+"_squeeze")(x_in)
    x = Reshape((1,layer_n),name=name+"_reshape")(x)
    x = Conv1D(layer_n//rate, kernel_size=1,strides=1, name=name+"_reduce")(x)
    x = Activation("relu",name=name+"_relu")(x)
    x = Conv1D(layer_n, kernel_size=1,strides=1, name=name+"_expand")(x)
    x = Activation("sigmoid",name=name+"_sigmoid")(x)
    x = Multiply(name=name+"_excite")([x_in, x])
    return x

def resblock_1d(x, out_ch, kernel, stride, name, bias=True, use_se=True, shortcut_conv=True):
    inputs = x
    x = cbr_1d(x, out_ch, kernel, stride, name+"_cbr0", bias)
    x = cbr_1d(x, out_ch, kernel, 1, name+"_cbr1", bias)
    if use_se:
        x = se_1d(x, out_ch, rate=4, name=name+"_se")
    #x_in = cbr(inputs, out_ch, 1, stride, name+"_cbr_shortcut", bias)
    # if stride!=1:
    if shortcut_conv:
        inputs = Conv1D(out_ch, kernel_size=1, strides=stride, padding="same", name=name+"_shortcut")(inputs)
    x = Add()([x, inputs])
    return x

def build_1d_model(input_shape=(7168, 4), num_cls=139):

    def diffabs_features(feature_1ch):
         # anglezのみdiffをとる
        x_diff_1 = Lambda(lambda x: tf.roll(x, shift=1, axis=1) - x)(feature_1ch)
        x_diff_2 = Lambda(lambda x: tf.roll(x, shift=-1, axis=1) - x)(feature_1ch)
        x_diff = Lambda(lambda x: tf.math.abs(x[0]) + tf.math.abs(x[1]))([x_diff_1, x_diff_2])
        # x = Concatenate()([x, x_diff])
        return x_diff
    
    def normalize(x):
        x = x - tf.reduce_mean(x, axis=1, keepdims=True)
        x = x / (tf.math.reduce_std(x, axis=1, keepdims=True)+1e-7)
        return x


    def positioanl_encode(day_feature):
        """
        0-1(0-24) -> sin, cos
        """
        def positional_encode_layer(x, min_timescale=4, max_timescale=24):
            outputs = []
            for omega in [1, 24]:#, 24*4]: #, 24*4, 24*4*4]:
                sin = tf.math.sin(x * 2.0 * math.pi * omega)
                cos = tf.math.cos(x * 2.0 * math.pi * omega)
                outputs.append(sin)
                outputs.append(cos)
            outputs = tf.concat(outputs, axis=-1)
            return outputs
        day_feature = Lambda(positional_encode_layer)(day_feature)
        return day_feature




    inputs = Input(input_shape, name="inputs")
    
    anglez = Lambda(lambda x: x[...,1:2])(inputs)
    # anglez = normalize(anglez)
    x_dabs = diffabs_features(anglez)
    day_feature = Lambda(lambda x: x[...,0:1])(inputs)
    day_feature = positioanl_encode(day_feature)
    x = Concatenate(axis=-1)([inputs, x_dabs, day_feature])
    # dropout
    x = Dropout(0.1, noise_shape=[1,1,None])(x) # input_shape[-1]+5
    # x = Dropout(0.1, noise_shape=[None,1,None])(x)


    # x = Lambda(lambda x: tf.concat([x[0][..., 1:2], x[0][..., -1:], x[1]], axis=-1))([inputs, x_dabs])

    # x = inputs
    # block_channels = [16, 16, 32, 64, 128, 256, 512, 512, 512, 1024, 1024]
    block_channels = [32, 32, 64, 64, 128, 256, 512, 512, 512, 1024, 1024]
    shortcuts = []
    x = Conv1D(block_channels[0], kernel_size=3, strides=1, padding="same", activation="relu", name="stem_conv")(x)
    shortcuts.append(x)

    # a_ave = anglez
    # a_max = anglez
    for i, ch in enumerate(block_channels[1:]):
        # if i > 0:
        #     # x_base = AveragePooling1D(pool_size=2)(x_base)
        #     a_ave = AveragePooling1D(pool_size=2)(a_ave)
        #     a_max = MaxPooling1D(pool_size=2)(a_max)
            
        #     if i > 1 and i%3==0:
        #         # x_ds = Conv1D(block_channels[0], kernel_size=3, strides=1, padding="same", activation="relu", name=f"stem_conv_next_{i}")(x_base)
        #         # x = Concatenate()([x, x_ds])

        #         a_ave_dabs = diffabs_features(a_ave)
        #         a_max_dabs = diffabs_features(a_max)
        #         a_features = Concatenate()([a_ave, a_max, a_ave_dabs, a_max_dabs])
        #         a_features = Conv1D(block_channels[0], kernel_size=3, strides=1, padding="same", activation="relu", name=f"stem_conv_next_{i}")(a_features)
        #         x = Concatenate()([x, a_features])

        x = resblock_1d(x, ch, 3, 2, f"resblock_{i}_down1")
        # if i>=5:
        #     x = resblock_1d(x, ch, 3, 1, f"resblock_{i}_down2", shortcut_conv=False)
        shortcuts.append(x)
    
    # x_cls = resblock_1d(x, ch, 5, 1, f"resblock_bottleneck_{i}", shortcut_conv=False)
    # x_cls = GlobalAveragePooling1D(name="gap")(x_cls)
    # x_cls = Dropout(0.2)(x_cls)
    # x_cls_emb = Dense(64, activation="relu", name="embedding_person")(x_cls)
    # x_cls = Dense(num_cls, activation="softmax", name="out_id_class")(x_cls_emb)

    # upsample UNet
    for i, ch in enumerate(block_channels[-2::-1]):
        x = UpSampling1D(size=2)(x)
        x = Concatenate()([x, shortcuts[-i-2]])
        x = resblock_1d(x, ch, 3, 1, f"resblock_{i}_up1")
    
    x_s = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x_sleepawake = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv")(x_s)
    x_sleepawake = Activation("sigmoid", name="out_state")(x_sleepawake)

    x_n = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x_nan = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_nan")(x_n)
    x_nan = Activation("sigmoid", name="out_state_nan")(x_nan)

    """
    # second
    x_stopgrad = Lambda(lambda x: tf.stop_gradient(x))(x_sleepawake)
    x_sa = x_stopgrad
    shortcuts_2 = []
    x_2 = Concatenate()([x_sa, shortcuts[0]])
    shortcuts_2.append(x_2)
    for i, sc in enumerate(shortcuts[1:]):
        x_sa = AveragePooling1D(pool_size=2)(x_sa)
        x_2 = Concatenate()([x_sa, sc])
        shortcuts_2.append(x_2)
    for i, ch in enumerate(block_channels[-2::-1]):
        x_sa = UpSampling1D(size=2)(x_sa)
        x_sa = Concatenate()([x_sa, shortcuts_2[-i-2]])
        x_sa = resblock_1d(x_sa, ch, 3, 1, f"resblock_{i}_up2")
    x = x_sa
    """
    # x_stopgrad = Lambda(lambda x: tf.stop_gradient(x))(x_sleepawake)
    # abs_feat = diffabs_features(x_stopgrad)
    # x_sw = Concatenate()([x, abs_feat])

    # x_sw = resblock_1d(x, 32, 5, 2, f"resblock_{i}_down1_sw")
    # x_sw = resblock_1d(x_sw, 32, 5, 1, f"resblock_{i}_down2_sw")
    # x_sw = resblock_1d(x_sw, 32, 5, 1, f"resblock_{i}_down3_sw")
    # x_sw = UpSampling1D(size=2)(x_sw)

    x_sw = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x_switch = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
    x_switch = Activation("sigmoid", name="out_event")(x_switch)
    
    outputs = [x_sleepawake, x_switch, x_nan]
    model = Model(inputs, outputs)
    loss = {"out_state": bce_loss_w_mask,
            "out_state_nan": bce_loss_w_mask,
            "out_event": bce_loss_switch_w_mask,
            }
    loss_weights = {"out_state": 1.,
                    "out_state_nan": 1.,
                    "out_event": 0.25,
                    }
    metrics = {"out_state": [accuracy_w_mask]}
    return model, loss, loss_weights, metrics

class TSM1D(tf.keras.layers.Layer):
    """TemporalShiftModule layer."""

    def __init__(self, num_shift, output_filters, name=None, forward=0.125, backward=0.125):
        super().__init__(name=name)
        #print(sequence_length)
        self.num_shift = num_shift
        self.forward = forward
        self.backward = backward
        self.output_filters = output_filters
        

    def call(self, inputs):
        batch, sequence_length, channel = tf.unstack(tf.shape(inputs))
        #print(batch, height, width, channel)
        # reshaped = tf.reshape(inputs, [-1, self.sequence_length, height, width, self.output_filters])
        #reshaped = tf.reshape(inputs, [int(tf.shape(inputs)[0]//self.sequence_length), 
        #                               self.sequence_length, tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]])
        forward_ch = tf.cast(self.forward * tf.cast(channel, tf.float32), tf.int32)
        backward_ch = tf.cast(self.backward * tf.cast(channel, tf.float32), tf.int32)
        forward = tf.roll(inputs[...,:forward_ch], shift=self.num_shift, axis=1)
        forward = tf.pad(forward[:,self.num_shift:], paddings=[[0,0],[self.num_shift,0],[0,0]], mode="CONSTANT", constant_values=0.)
        stay = inputs[...,forward_ch:-backward_ch]
        backward = tf.roll(inputs[...,-backward_ch:], shift=-self.num_shift, axis=1)
        backward = tf.pad(forward[:,:-self.num_shift], paddings=[[0,0],[0,self.num_shift],[0,0]], mode="CONSTANT", constant_values=0.)
        outputs = tf.concat([forward, stay, backward], axis=-1)
        outputs = tf.reshape(outputs, [batch, sequence_length, self.output_filters])
        return outputs



def build_1d_model_multiclass(input_shape=(7168, 4), num_cls=11, D=True, Ws=[1,24], SSLmodel=False, outstride=1, base_kernel=3):

    def diffabs_features(feature_1ch):
         # anglezのみdiffをとる
        x_diff_1 = Lambda(lambda x: tf.roll(x, shift=1, axis=1) - x)(feature_1ch)
        x_diff_2 = Lambda(lambda x: tf.roll(x, shift=-1, axis=1) - x)(feature_1ch)
        x_diff = Lambda(lambda x: tf.math.abs(x[0]) + tf.math.abs(x[1]))([x_diff_1, x_diff_2])
        # x = Concatenate()([x, x_diff])
        return x_diff

    def angle_features(feature_1ch):
         # anglezのみdiffをとる
        x_diff = Lambda(lambda x: tf.roll(x, shift=1, axis=1) - x)(feature_1ch)
        x_diff_log1p = Lambda(lambda x: tf.math.log1p(tf.math.abs(x)) / 5.)(x_diff)
        x_diff_100 = Lambda(lambda x: tf.clip_by_value(x * 100., -1, 1))(x_diff)
        x_zero = Lambda(lambda x: tf.cast(tf.math.abs(x) < 1e-5, tf.float32))(x_diff)
        x = Concatenate()([x_diff, x_diff_log1p, x_diff_100, x_zero])
        return x_diff
    
    def normalize(x):
        x = x - tf.reduce_mean(x, axis=1, keepdims=True)
        x = x / (tf.math.reduce_std(x, axis=1, keepdims=True)+1e-7)
        return x


    def positioanl_encode(day_feature, omegas=[1, 24]):
        """
        0-1(0-24) -> sin, cos
        """
        def positional_encode_layer(x, min_timescale=4, max_timescale=24):
            outputs = []
            for omega in omegas:#, 24*4]: #, 24*4, 24*4*4]:
                sin = tf.math.sin(x * 2.0 * math.pi * omega)
                cos = tf.math.cos(x * 2.0 * math.pi * omega)
                outputs.append(sin)
                # outputs.append(cos)
            outputs = tf.concat(outputs, axis=-1)
            return outputs
        day_feature = Lambda(positional_encode_layer)(day_feature)
        return day_feature

    def ssl_loss(y_true, y_pred):
        # y_true はとりあえずダミー。ほんとはスプリットしなくても合致サンプル組はあるはず…。
        # y_pred は [2xb, 256]なので、前半後半にわけて、前半と後半でcos類似度をとる
        y_pred_1, y_pred_2 = tf.split(y_pred, 2, axis=0)
        y_pred_1 = tf.math.l2_normalize(y_pred_1, axis=-1)[:, tf.newaxis, :]
        y_pred_2 = tf.math.l2_normalize(y_pred_2, axis=-1)[tf.newaxis, :, :]
        cos_sim_matrix = y_pred_1 * y_pred_2 # [b, b, 256]
        cos_sim_matrix = 10 * tf.reduce_sum(cos_sim_matrix, axis=-1) # [b, b] # 10 is temperature
        # sigmoid
        cos_sim_matrix = tf.math.sigmoid(cos_sim_matrix)

        y_true = tf.eye(tf.shape(cos_sim_matrix)[0], dtype=tf.float32)
        loss = bce_loss_w_mask(y_true, cos_sim_matrix)
        return loss

    block_channels = [32, 32, 64, 64, 128, 256, 512, 512, 512, 1024, 1024] + [1024]
    shortcuts = []

    inputs = Input(input_shape, name="inputs") # b, l, ch
    if SSLmodel:
        inputs_split = Lambda(lambda x: tf.concat(tf.split(x, 2, axis=1), axis=0), name="inputs_split")(inputs) # 2b, l/2, ch
    else:
        inputs_split = inputs

    # ['daily_step', 'anglez', 'anglez_simpleerror', 'anglez_simpleerror_span', 'anglez_nanexist', 'anglez_daystd', 'anglez_daymean', "anglez_daycounter", "anglez_nancounter", 'enmo']
    
    day_feature = Lambda(lambda x: x[...,0:1])(inputs_split)
    anglez = Lambda(lambda x: x[...,1:2])(inputs_split)
    enmo = Lambda(lambda x: x[...,9:10])(inputs_split)

    day_feature_pe = positioanl_encode(day_feature, omegas=Ws)
    anglez_dabs = diffabs_features(anglez)
    anglez_feat = angle_features(anglez)
    enmo_scaled = Lambda(lambda x: tf.clip_by_value(x*25, 0, 1))(enmo)
    enmo_under0001 = Lambda(lambda x: tf.cast(x<0.001, tf.float32))(enmo)
    
    stem_inputs_angle = Concatenate(axis=-1)([anglez, anglez_dabs, anglez_feat])
    stem_inputs_enmo = Concatenate(axis=-1)([enmo, enmo_scaled, enmo_under0001])
    stem_inputs_days = day_feature_pe
    stem_inputs_others = Lambda(lambda x: x[...,2:9])(inputs_split)


    # SplitStem
    stem_ch = 24 + 16
    stem_inputs_angle = Conv1D(stem_ch, kernel_size=3, strides=1, padding="same", activation="relu", name="stem_conv_angle")(stem_inputs_angle)
    stem_inputs_enmo = Conv1D(stem_ch, kernel_size=3, strides=1, padding="same", activation="relu", name="stem_conv_enmo")(stem_inputs_enmo)
    stem_inputs_others = Conv1D(stem_ch, kernel_size=3, strides=1, padding="same", activation="relu", name="stem_conv_others")(stem_inputs_others)
    # sum all stem outputs
    
    stem_inputs_days = Conv1D(stem_ch, kernel_size=3, strides=1, padding="same", activation="relu", name="stem_conv_days")(stem_inputs_days)
    x = [
        stem_inputs_angle,
        stem_inputs_enmo,
        stem_inputs_days,
        stem_inputs_others,
    ]
    x = Lambda(lambda x: x[0]/3 + x[1]/3 + x[2]/3 + x[3]/3, name="stem_inputs")(x)

    if D==True:
        x = Dropout(0.25, noise_shape=[1,1,None])(x)



    """

    # enmo = Lambda(lambda x: x[...,9:10])(inputs_split)
    # scaled_enmo = Lambda(lambda x: tf.clip_by_value(x*5, 0, 1))(enmo)
    # anglez_scaled_enmo = Concatenate()([anglez, scaled_enmo])
    # mean_anglez_scaled_enmo = anglez_scaled_enmo
    # for i in range(6):
    #     mean_anglez_scaled_enmo = AveragePooling1D(pool_size=2)(mean_anglez_scaled_enmo)
    # for i in range(6):
    #     mean_anglez_scaled_enmo = UpSampling1D(size=2)(mean_anglez_scaled_enmo)
    # anglez_scaled_enmo_diff = Lambda(lambda x: x[1] - x[0])([mean_anglez_scaled_enmo, anglez_scaled_enmo])



    
    x = Concatenate(axis=-1)([inputs_split, x_dabs])
    
    if D==True:
        x = Dropout(0.25, noise_shape=[1,1,None])(x) # input_shape[-1]+5
        # x = Dropout(0.1+0.05, noise_shape=[1,1,None])(x) # input_shape[-1]+5
    else:
        print("no dropout!!!!!!!!!")

    x = Concatenate()([x, day_feature])

    # block_channels = [32, 32, 64, 64, 128, 256, 512, 512, 512, 1024, 1024]
    # block_channels = block_channels
    # shortcuts = []
    x = Conv1D(block_channels[0], kernel_size=3, strides=1, padding="same", activation="relu", name="stem_conv")(x)

    """




    shortcuts.append(x)

    # x_feat_ave = Concatenate()([stem_inputs_angle, stem_inputs_enmo])
    # x_feat_max = x_feat_ave
    for i, ch in enumerate(block_channels[1:]):
        # if i > 0:
        #     x_feat_ave = AveragePooling1D(pool_size=2)(x_feat_ave)
        #     x_feat_max = MaxPooling1D(pool_size=2)(x_feat_max)
        #     if i > 1 and i%3==0:
        #         x_sub = Concatenate()([x_feat_ave, x_feat_max])
        #         x_sub = Conv1D(block_channels[0], kernel_size=3, strides=1, padding="same", activation="relu", name=f"stem_conv_sub_{i}_0")(x_sub)
        #         x_sub = Conv1D(block_channels[0], kernel_size=3, strides=1, padding="same", activation="relu", name=f"stem_conv_sub_{i}_1")(x_sub)
        #         x = Concatenate()([x, x_sub]) # まだテストしてない★
            
        #     if i > 1 and i%3==0:
        #         # x_ds = Conv1D(block_channels[0], kernel_size=3, strides=1, padding="same", activation="relu", name=f"stem_conv_next_{i}")(x_base)
        #         # x = Concatenate()([x, x_ds])

        #         a_ave_dabs = diffabs_features(a_ave)
        #         a_max_dabs = diffabs_features(a_max)
        #         a_features = Concatenate()([a_ave, a_max, a_ave_dabs, a_max_dabs])
        #         a_features = Conv1D(block_channels[0], kernel_size=3, strides=1, padding="same", activation="relu", name=f"stem_conv_next_{i}")(a_features)
        #         x = Concatenate()([x, a_features])

        x = resblock_1d(x, ch, base_kernel, 2, f"resblock_{i}_down1")
        # if i in range(7):
        #     # 2**4 = 16, 2**5 = 32, 2**6 = 64, 2**7 = 128, 2**8 = 256, 2**9 = 512, 2**10 = 1024
        #     # 720(60min)とか360(30min)のシフトに興味がある
        #     if i in [0,1,2]:
        #         x = TSM1D(num_shift=90, output_filters=ch, name=f"tsm_{i}", forward=0.1, backward=0.1)(x)
        #     elif i in [3,4]:
        #         x = TSM1D(num_shift=45, output_filters=ch, name=f"tsm_{i}", forward=0.1, backward=0.1)(x)
        #     else:
        #         x = TSM1D(num_shift=11, output_filters=ch, name=f"tsm_{i}", forward=0.1, backward=0.1)(x)

        # x = resblock_1d(x, ch, 3, 1, f"resblock_{i}_down2") # , shortcut_conv=False)
        # x = Dropout(0.1)(x)
        shortcuts.append(x)
    
    if SSLmodel:
        x_emb = resblock_1d(x, ch, 5, 1, f"resblock_bottleneck_{i}", shortcut_conv=False)
        x_emb = GlobalAveragePooling1D(name="gap")(x_emb)
        x_emb = Dense(256, activation="linear", name="embedding_outputs")(x_emb)
        # normalize
        # x_emb = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1), "embedding_outputs")(x_emb)
        
        model = Model(inputs, x_emb)
        loss = {"embedding_outputs": ssl_loss}
        loss_weights = {"embedding_outputs": 1.}
        metrics = {"embedding_outputs": [ssl_loss]}
        return model, loss, loss_weights, metrics




    def merge_day_feat(inputs, num_feat=32):
        num_days = 3
        x = inputs
        batch, time, feat = tf.unstack(tf.shape(x))
        x = tf.reshape(x, [num_days, batch//num_days, time, feat])
        x_mean = tf.reduce_mean(x, axis=0)
        x_max = tf.reduce_max(x, axis=0)
        outputs = []
        for i in range(num_days):
            x_base = x[i]
            x_concat = tf.concat([x_mean, x_max, x_base], axis=-1)
            x_concat = tf.reshape(x_concat, [batch//num_days, time, num_feat*3])
            outputs.append(x_concat)
        x = tf.concat(outputs, axis=0)
        return x
    
    #  = Lambda(merge_day_feat, arguments={"num_feat": block_channels[-1]})(x)
    # for i in range(len(shortcuts)):
    #     shortcuts[i] = Lambda(merge_day_feat, arguments={"num_feat": block_channels[i]})(shortcuts[i])
    # shortcuts = aggregation_1d(shortcuts, output_layer_n=12) # aggregation not needed
    # upsample UNet
    upsample_blocks = block_channels[-2::-1]
    for i, ch in enumerate(upsample_blocks):
        x = UpSampling1D(size=2)(x)
        sc = shortcuts[-i-2]
        # sc = resblock_1d(sc, ch, 3, 1, f"resblock_{i}_up1sc")
        
        # if i < 5:
        #     sc = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(ch, return_sequences=True))(sc)
        x = Concatenate()([x, sc])
            
        x = resblock_1d(x, ch, base_kernel, 1, f"resblock_{i}_up1")

        # x = cbr_1d(x, ch, 3, 1, f"up_cbr_{i}")

        if outstride>1:
            if outstride!=4 and outstride!=2:
                raise NotImplementedError
            else:
                if i == len(upsample_blocks)-3 and outstride==4:
                    break
                elif i == len(upsample_blocks)-2 and outstride==2:
                    break

    
    x_s = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    # x_s = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(x_s)
    x_sleepawake = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv")(x_s)
    x_sleepawake = Activation("sigmoid", name="out_state")(x_sleepawake)

    x_n = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    # x_n = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(x_n)
    x_nan = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_nan")(x_n)
    x_nan = Activation("sigmoid", name="out_state_nan")(x_nan)

    """
    # second
    x_stopgrad = Lambda(lambda x: tf.stop_gradient(x))(x_sleepawake)
    x_sa = x_stopgrad
    shortcuts_2 = []
    x_2 = Concatenate()([x_sa, shortcuts[0]])
    shortcuts_2.append(x_2)
    for i, sc in enumerate(shortcuts[1:]):
        x_sa = AveragePooling1D(pool_size=2)(x_sa)
        x_2 = Concatenate()([x_sa, sc])
        shortcuts_2.append(x_2)
    for i, ch in enumerate(block_channels[-2::-1]):
        x_sa = UpSampling1D(size=2)(x_sa)
        x_sa = Concatenate()([x_sa, shortcuts_2[-i-2]])
        x_sa = resblock_1d(x_sa, ch, 3, 1, f"resblock_{i}_up2")
    x = x_sa
    """
    # x_stopgrad = Lambda(lambda x: tf.stop_gradient(x))(x_sleepawake)
    # abs_feat = diffabs_features(x_stopgrad)
    # x_sw = Concatenate()([x, abs_feat])

    # x_sw = resblock_1d(x, 32, 5, 2, f"resblock_{i}_down1_sw")
    # x_sw = resblock_1d(x_sw, 32, 5, 1, f"resblock_{i}_down2_sw")
    # x_sw = resblock_1d(x_sw, 32, 5, 1, f"resblock_{i}_down3_sw")
    # x_sw = UpSampling1D(size=2)(x_sw)
    manyouts = True
    if not manyouts:
        x_sw = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(x)
        x_switch_s10 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s10", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
        x_switch_s10 = Activation("sigmoid", name="out_event_s10")(x_switch_s10)

        x_switch_s8 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s8", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
        x_switch_s8 = Activation("sigmoid", name="out_event_s8")(x_switch_s8)

        outputs = [x_sleepawake, x_switch_s10, x_switch_s8, x_nan]
        loss = {"out_state": bce_loss_w_mask,
            "out_state_nan": bce_loss_w_mask,
            "out_event_s10": binary_focal_loss_switch,
            "out_event_s8": binary_focal_loss_switch,
            }
        loss_weights = {"out_state": 1.,
                        "out_state_nan": 1.,
                        "out_event_s10": 0.25,
                        "out_event_s8": 0.25,
                        }

    else:
        x_sw = Conv1D(32, kernel_size=3, strides=1, activation="relu", padding="same")(x)
        # x_sw = Conv1D(32, kernel_size=3, strides=1, activation="relu", padding="same")(x_sw)

        x_switch_s10p = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s10p", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
        x_switch_s10p = Activation("sigmoid", name="out_event_s10p")(x_switch_s10p)

        x_switch_s10 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s10", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
        x_switch_s10 = Activation("sigmoid", name="out_event_s10")(x_switch_s10)

        x_switch_s8 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s8", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
        x_switch_s8 = Activation("sigmoid", name="out_event_s8")(x_switch_s8)

        x_switch_s6 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s6", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
        x_switch_s6 = Activation("sigmoid", name="out_event_s6")(x_switch_s6)

        x_switch_s4 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s4", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
        x_switch_s4 = Activation("sigmoid", name="out_event_s4")(x_switch_s4)

        x_switch_s2 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s2", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
        x_switch_s2 = Activation("sigmoid", name="out_event_s2")(x_switch_s2)

        outputs = [x_sleepawake, x_switch_s10p, x_switch_s10, x_switch_s8, x_switch_s6, x_switch_s4, x_switch_s2, x_nan]
        loss = {"out_state": bce_loss_w_mask, # state_loss_focal,
            "out_state_nan": bce_loss_w_mask,
            "out_event_s10p": binary_focal_loss_switch,
            "out_event_s10": binary_focal_loss_switch,
            "out_event_s8": binary_focal_loss_switch,
            "out_event_s6": binary_focal_loss_switch,
            "out_event_s4": binary_focal_loss_switch,
            "out_event_s2": binary_focal_loss_switch,
            }
        loss_weights = {"out_state": 1.,
                        "out_state_nan": 1.,
                        "out_event_s10p": 0.25,
                        "out_event_s10": 0.35,
                        "out_event_s8": 0.35,
                        "out_event_s6": 0.25,
                        "out_event_s4": 0.25,
                        "out_event_s2": 0.25,
                        }

    """
    # HourGlass

    outputs_1st = outputs
    loss_1st = loss
    loss_weights_1st = loss_weights

        
    # すべての出力をstop_gradient
    second_inputs = [x_stem]# + outputs
    for out in outputs:
        second_inputs.append(Lambda(lambda x: tf.stop_gradient(x))(out))
    x = Concatenate()(second_inputs)
    shortcuts = []
    shortcuts.append(x)
    for i, ch in enumerate(block_channels[1:]):
        x = resblock_1d(x, ch, 3, 2, f"resblock_{i}_down1_2ndUnet")
        shortcuts.append(x)
    for i, ch in enumerate(upsample_blocks):
        x = UpSampling1D(size=2)(x)
        sc = shortcuts[-i-2]
        x = Concatenate()([x, sc])
        x = resblock_1d(x, ch, 3, 1, f"resblock_{i}_up1_2ndUnet")
        if outstride>1:
            if outstride!=4 and outstride!=2:
                raise NotImplementedError
            else:
                if i == len(upsample_blocks)-3 and outstride==4:
                    break
                elif i == len(upsample_blocks)-2 and outstride==2:
                    break
    
    x_s = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x_sleepawake = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_2ndUnet")(x_s)
    x_sleepawake = Activation("sigmoid", name="out_state_2ndUnet")(x_sleepawake)

    x_n = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x_nan = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_nan_2ndUnet")(x_n)
    x_nan = Activation("sigmoid", name="out_state_nan_2ndUnet")(x_nan)

    x_sw = Conv1D(32, kernel_size=3, strides=1, activation="relu", padding="same")(x)

    x_switch_s10p = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s10p_2ndUnet", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
    x_switch_s10p = Activation("sigmoid", name="out_event_s10p_2ndUnet")(x_switch_s10p)

    x_switch_s10 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s10_2ndUnet", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
    x_switch_s10 = Activation("sigmoid", name="out_event_s10_2ndUnet")(x_switch_s10)

    x_switch_s8 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s8_2ndUnet", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
    x_switch_s8 = Activation("sigmoid", name="out_event_s8_2ndUnet")(x_switch_s8)

    x_switch_s6 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s6_2ndUnet", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
    x_switch_s6 = Activation("sigmoid", name="out_event_s6_2ndUnet")(x_switch_s6)

    x_switch_s4 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s4_2ndUnet", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
    x_switch_s4 = Activation("sigmoid", name="out_event_s4_2ndUnet")(x_switch_s4)

    x_switch_s2 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s2_2ndUnet", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
    x_switch_s2 = Activation("sigmoid", name="out_event_s2_2ndUnet")(x_switch_s2)

    outputs_2nd = [x_sleepawake, x_switch_s10p, x_switch_s10, x_switch_s8, x_switch_s6, x_switch_s4, x_switch_s2, x_nan]
    loss_2nd = {"out_state_2ndUnet": bce_loss_w_mask,
        "out_state_nan_2ndUnet": bce_loss_w_mask,
        "out_event_s10p_2ndUnet": binary_focal_loss_switch,
        "out_event_s10_2ndUnet": binary_focal_loss_switch,
        "out_event_s8_2ndUnet": binary_focal_loss_switch,
        "out_event_s6_2ndUnet": binary_focal_loss_switch,
        "out_event_s4_2ndUnet": binary_focal_loss_switch,
        "out_event_s2_2ndUnet": binary_focal_loss_switch,
        }
    loss_weights_2nd = {"out_state_2ndUnet": 1.,
                    "out_state_nan_2ndUnet": 1.,
                    "out_event_s10p_2ndUnet": 0.25,
                    "out_event_s10_2ndUnet": 0.35,
                    "out_event_s8_2ndUnet": 0.35,
                    "out_event_s6_2ndUnet": 0.25,
                    "out_event_s4_2ndUnet": 0.25,
                    "out_event_s2_2ndUnet": 0.25,
                    }
    outputs = outputs_1st + outputs_2nd
    loss = {**loss_1st, **loss_2nd}
    loss_weights = {**loss_weights_1st, **loss_weights_2nd}
    """



    def cls_to_expect(inputs):
        s10, s8 = inputs
        return (s10 * 1 + s8 * 0.5) / 1.6

    # hourglass
    # ensemble_outputs = []
    # for out_1, out_2 in zip(outputs_1st, outputs_2nd):
    #     out = Lambda(lambda x: (x[0] + x[1]) / 2)([out_1, out_2])
    #     ensemble_outputs.append(out)


    # もともと
    event_outputs = [x_switch_s10p, x_switch_s10, x_switch_s8, x_switch_s6, x_switch_s4, x_switch_s2]
    # hourglass
    # event_outputs = ensemble_outputs[1:-1]

    x_switch_expect = Lambda(lambda x: tf.concat(x, axis=-1), name="infer_out_event")(event_outputs)

    # もともと
    outputs_infer = [x_sleepawake, x_switch_expect, x_nan]
    # hourglass
    # outputs_infer = ensemble_outputs[0:1] + [x_switch_expect] + ensemble_outputs[-1:]


    model = Model(inputs, outputs)
    model_infer = Model(inputs, outputs_infer)
    
    metrics = None # {"out_state": [accuracy_w_mask]}

    return model, model_infer, loss, loss_weights, metrics



def build_1d_model_multiclass_controledstride(input_shape=(7168, 4), num_cls=11, D=True, Ws=[1,24], SSLmodel=False, outstride=1, model_type="v2"):

    def diffabs_features(feature_1ch):
         # anglezのみdiffをとる
        x_diff_1 = Lambda(lambda x: tf.roll(x, shift=1, axis=1) - x)(feature_1ch)
        x_diff_2 = Lambda(lambda x: tf.roll(x, shift=-1, axis=1) - x)(feature_1ch)
        x_diff = Lambda(lambda x: tf.math.abs(x[0]) + tf.math.abs(x[1]))([x_diff_1, x_diff_2])
        # x = Concatenate()([x, x_diff])
        return x_diff

    def angle_features(feature_1ch):
         # anglezのみdiffをとる
        x_diff = Lambda(lambda x: tf.roll(x, shift=1, axis=1) - x)(feature_1ch)
        x_diff_log1p = Lambda(lambda x: tf.math.log1p(tf.math.abs(x)) / 5.)(x_diff)
        x_diff_100 = Lambda(lambda x: tf.clip_by_value(x * 100., -1, 1))(x_diff)
        x_zero = Lambda(lambda x: tf.cast(tf.math.abs(x) < 1e-5, tf.float32))(x_diff)
        x = Concatenate()([x_diff, x_diff_log1p, x_diff_100, x_zero])
        return x_diff
    
    def normalize(x):
        x = x - tf.reduce_mean(x, axis=1, keepdims=True)
        x = x / (tf.math.reduce_std(x, axis=1, keepdims=True)+1e-7)
        return x


    def positioanl_encode(day_feature, omegas=[1, 24]):
        """
        0-1(0-24) -> sin, cos
        """
        def positional_encode_layer(x, min_timescale=4, max_timescale=24):
            outputs = []
            for omega in omegas:#, 24*4]: #, 24*4, 24*4*4]:
                sin = tf.math.sin(x * 2.0 * math.pi * omega)
                cos = tf.math.cos(x * 2.0 * math.pi * omega)
                outputs.append(sin)
                # outputs.append(cos)
            outputs = tf.concat(outputs, axis=-1)
            return outputs
        day_feature = Lambda(positional_encode_layer)(day_feature)
        return day_feature

    def ssl_loss(y_true, y_pred):
        # y_true はとりあえずダミー。ほんとはスプリットしなくても合致サンプル組はあるはず…。
        # y_pred は [2xb, 256]なので、前半後半にわけて、前半と後半でcos類似度をとる
        y_pred_1, y_pred_2 = tf.split(y_pred, 2, axis=0)
        y_pred_1 = tf.math.l2_normalize(y_pred_1, axis=-1)[:, tf.newaxis, :]
        y_pred_2 = tf.math.l2_normalize(y_pred_2, axis=-1)[tf.newaxis, :, :]
        cos_sim_matrix = y_pred_1 * y_pred_2 # [b, b, 256]
        cos_sim_matrix = 10 * tf.reduce_sum(cos_sim_matrix, axis=-1) # [b, b] # 10 is temperature
        # sigmoid
        cos_sim_matrix = tf.math.sigmoid(cos_sim_matrix)

        y_true = tf.eye(tf.shape(cos_sim_matrix)[0], dtype=tf.float32)
        loss = bce_loss_w_mask(y_true, cos_sim_matrix)
        return loss

    # block_channels = [32, 32, 64, 96, 128, 256, 512, 512, 512, 1024, 1024, 1024]
    # strides = [1, 2, 2, 3, 1, 3, 5, 2, 2, 2, 2, 1]
    if model_type=="v2":
        print("model_type: v2")
        block_channels = [32, 32, 64, 96, 128, 256, 512, 512, 512, 1024, 1024, 1024]
        strides = [1, 2, 2, 3, 1, 1, 1, 3, 5, 4, 4, 1]
    elif model_type=="v3":
        print("model_type: v3")
        block_channels = [32, 32, 64, 96, 128, 256, 256, 256, 512, 1024, 1024, 1024, 1024]
        strides = [1, 2, 2, 1, 1, 3, 1, 1, 3, 5, 4, 4, 1]
    else:
        raise NotImplementedError
    shortcuts = []

    inputs = Input(input_shape, name="inputs") # b, l, ch
    if SSLmodel:
        inputs_split = Lambda(lambda x: tf.concat(tf.split(x, 2, axis=1), axis=0), name="inputs_split")(inputs) # 2b, l/2, ch
    else:
        inputs_split = inputs

    # ['daily_step', 'anglez', 'anglez_simpleerror', 'anglez_simpleerror_span', 'anglez_nanexist', 'anglez_daystd', 'anglez_daymean', "anglez_daycounter", "anglez_nancounter", 'enmo']
    
    day_feature = Lambda(lambda x: x[...,0:1])(inputs_split)
    anglez = Lambda(lambda x: x[...,1:2])(inputs_split)
    enmo = Lambda(lambda x: x[...,9:10])(inputs_split)

    day_feature_pe = positioanl_encode(day_feature, omegas=Ws)
    anglez_dabs = diffabs_features(anglez)
    anglez_feat = angle_features(anglez)
    enmo_scaled = Lambda(lambda x: tf.clip_by_value(x*25, 0, 1))(enmo)
    enmo_under0001 = Lambda(lambda x: tf.cast(x<0.001, tf.float32))(enmo)
    
    stem_inputs_angle = Concatenate(axis=-1)([anglez, anglez_dabs, anglez_feat])
    stem_inputs_enmo = Concatenate(axis=-1)([enmo, enmo_scaled, enmo_under0001])
    stem_inputs_days = day_feature_pe
    stem_inputs_others = Lambda(lambda x: x[...,2:9])(inputs_split)


    # SplitStem
    stem_ch = 24 + 16
    stem_inputs_angle = Conv1D(stem_ch, kernel_size=3, strides=1, padding="same", activation="relu", name="stem_conv_angle")(stem_inputs_angle)
    stem_inputs_enmo = Conv1D(stem_ch, kernel_size=3, strides=1, padding="same", activation="relu", name="stem_conv_enmo")(stem_inputs_enmo)
    stem_inputs_others = Conv1D(stem_ch, kernel_size=3, strides=1, padding="same", activation="relu", name="stem_conv_others")(stem_inputs_others)
    # sum all stem outputs
    
    stem_inputs_days = Conv1D(stem_ch, kernel_size=3, strides=1, padding="same", activation="relu", name="stem_conv_days")(stem_inputs_days)
    x = [
        stem_inputs_angle,
        stem_inputs_enmo,
        stem_inputs_days,
        stem_inputs_others,
    ]
    x = Lambda(lambda x: x[0]/3 + x[1]/3 + x[2]/3 + x[3]/3, name="stem_inputs")(x)

    if D==True:
        x = Dropout(0.25, noise_shape=[1,1,None])(x) # input_shape[-1]+5


    shortcuts.append(x)

    for i, ch in enumerate(block_channels[1:]):
        stride_ = strides[i+1]
        # if i > 0:
        #     x_feat_ave = AveragePooling1D(pool_size=2)(x_feat_ave)
        #     x_feat_max = MaxPooling1D(pool_size=2)(x_feat_max)
        #     if i > 1 and i%3==0:
        #         x_sub = Concatenate()([x_feat_ave, x_feat_max])
        #         x_sub = Conv1D(block_channels[0], kernel_size=3, strides=1, padding="same", activation="relu", name=f"stem_conv_sub_{i}_0")(x_sub)
        #         x_sub = Conv1D(block_channels[0], kernel_size=3, strides=1, padding="same", activation="relu", name=f"stem_conv_sub_{i}_1")(x_sub)
        #         x = Concatenate()([x, x_sub]) # まだテストしてない★
            
        #     if i > 1 and i%3==0:
        #         # x_ds = Conv1D(block_channels[0], kernel_size=3, strides=1, padding="same", activation="relu", name=f"stem_conv_next_{i}")(x_base)
        #         # x = Concatenate()([x, x_ds])

        #         a_ave_dabs = diffabs_features(a_ave)
        #         a_max_dabs = diffabs_features(a_max)
        #         a_features = Concatenate()([a_ave, a_max, a_ave_dabs, a_max_dabs])
        #         a_features = Conv1D(block_channels[0], kernel_size=3, strides=1, padding="same", activation="relu", name=f"stem_conv_next_{i}")(a_features)
        #         x = Concatenate()([x, a_features])

        x = resblock_1d(x, ch, 3, stride_, f"resblock_{i}_down1")
        # if i in range(7):
        #     # 2**4 = 16, 2**5 = 32, 2**6 = 64, 2**7 = 128, 2**8 = 256, 2**9 = 512, 2**10 = 1024
        #     # 720(60min)とか360(30min)のシフトに興味がある
        #     if i in [0,1,2]:
        #         x = TSM1D(num_shift=90, output_filters=ch, name=f"tsm_{i}", forward=0.1, backward=0.1)(x)
        #     elif i in [3,4]:
        #         x = TSM1D(num_shift=45, output_filters=ch, name=f"tsm_{i}", forward=0.1, backward=0.1)(x)
        #     else:
        #         x = TSM1D(num_shift=11, output_filters=ch, name=f"tsm_{i}", forward=0.1, backward=0.1)(x)

        # x = resblock_1d(x, ch, 3, 1, f"resblock_{i}_down2") # , shortcut_conv=False)
        # x = Dropout(0.1)(x)
        shortcuts.append(x)
    
    if SSLmodel:
        x_emb = resblock_1d(x, ch, 5, 1, f"resblock_bottleneck_{i}", shortcut_conv=False)
        x_emb = GlobalAveragePooling1D(name="gap")(x_emb)
        x_emb = Dense(256, activation="linear", name="embedding_outputs")(x_emb)
        # normalize
        # x_emb = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1), "embedding_outputs")(x_emb)
        
        model = Model(inputs, x_emb)
        loss = {"embedding_outputs": ssl_loss}
        loss_weights = {"embedding_outputs": 1.}
        metrics = {"embedding_outputs": [ssl_loss]}
        return model, loss, loss_weights, metrics




    def merge_day_feat(inputs, num_feat=32):
        num_days = 3
        x = inputs
        batch, time, feat = tf.unstack(tf.shape(x))
        x = tf.reshape(x, [num_days, batch//num_days, time, feat])
        x_mean = tf.reduce_mean(x, axis=0)
        x_max = tf.reduce_max(x, axis=0)
        outputs = []
        for i in range(num_days):
            x_base = x[i]
            x_concat = tf.concat([x_mean, x_max, x_base], axis=-1)
            x_concat = tf.reshape(x_concat, [batch//num_days, time, num_feat*3])
            outputs.append(x_concat)
        x = tf.concat(outputs, axis=0)
        return x
    
    #  = Lambda(merge_day_feat, arguments={"num_feat": block_channels[-1]})(x)
    # for i in range(len(shortcuts)):
    #     shortcuts[i] = Lambda(merge_day_feat, arguments={"num_feat": block_channels[i]})(shortcuts[i])
    # shortcuts = aggregation_1d(shortcuts, output_layer_n=12) # aggregation not needed
    # upsample UNet
    upsample_blocks = block_channels[-2::-1]
    upsample_strides = strides[-1::-1]
    for i, ch in enumerate(upsample_blocks):
        stride_ = upsample_strides[i]
        x = UpSampling1D(size=stride_)(x)
        sc = shortcuts[-i-2]
        # if model_type=="v3":
        #     sc = Conv1D(ch//2, kernel_size=3, strides=1, padding="same", activation="relu", name=f"resblock_{i}_up1sc")(sc)
        # sc = resblock_1d(sc, ch, 3, 1, f"resblock_{i}_up1sc")
        
        # if i < 5:
        #     sc = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(ch, return_sequences=True))(sc)
        x = Concatenate()([x, sc])
            
        x = resblock_1d(x, ch, 3, 1, f"resblock_{i}_up1")

        # x = cbr_1d(x, ch, 3, 1, f"up_cbr_{i}")

        if outstride>1:
            if outstride!=4 and outstride!=2:
                raise NotImplementedError
            else:
                if i == len(upsample_blocks)-3 and outstride==4:
                    break
                elif i == len(upsample_blocks)-2 and outstride==2:
                    break

    
    x_s = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    # x_s = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(x_s)
    x_sleepawake = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv")(x_s)
    x_sleepawake = Activation("sigmoid", name="out_state")(x_sleepawake)

    x_n = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    # x_n = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(x_n)
    x_nan = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_nan")(x_n)
    x_nan = Activation("sigmoid", name="out_state_nan")(x_nan)

    """
    # second
    x_stopgrad = Lambda(lambda x: tf.stop_gradient(x))(x_sleepawake)
    x_sa = x_stopgrad
    shortcuts_2 = []
    x_2 = Concatenate()([x_sa, shortcuts[0]])
    shortcuts_2.append(x_2)
    for i, sc in enumerate(shortcuts[1:]):
        x_sa = AveragePooling1D(pool_size=2)(x_sa)
        x_2 = Concatenate()([x_sa, sc])
        shortcuts_2.append(x_2)
    for i, ch in enumerate(block_channels[-2::-1]):
        x_sa = UpSampling1D(size=2)(x_sa)
        x_sa = Concatenate()([x_sa, shortcuts_2[-i-2]])
        x_sa = resblock_1d(x_sa, ch, 3, 1, f"resblock_{i}_up2")
    x = x_sa
    """
    # x_stopgrad = Lambda(lambda x: tf.stop_gradient(x))(x_sleepawake)
    # abs_feat = diffabs_features(x_stopgrad)
    # x_sw = Concatenate()([x, abs_feat])

    # x_sw = resblock_1d(x, 32, 5, 2, f"resblock_{i}_down1_sw")
    # x_sw = resblock_1d(x_sw, 32, 5, 1, f"resblock_{i}_down2_sw")
    # x_sw = resblock_1d(x_sw, 32, 5, 1, f"resblock_{i}_down3_sw")
    # x_sw = UpSampling1D(size=2)(x_sw)
    manyouts = True
    if not manyouts:
        x_sw = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(x)
        x_switch_s10 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s10", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
        x_switch_s10 = Activation("sigmoid", name="out_event_s10")(x_switch_s10)

        x_switch_s8 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s8", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
        x_switch_s8 = Activation("sigmoid", name="out_event_s8")(x_switch_s8)

        outputs = [x_sleepawake, x_switch_s10, x_switch_s8, x_nan]
        loss = {"out_state": bce_loss_w_mask,
            "out_state_nan": bce_loss_w_mask,
            "out_event_s10": binary_focal_loss_switch,
            "out_event_s8": binary_focal_loss_switch,
            }
        loss_weights = {"out_state": 1.,
                        "out_state_nan": 1.,
                        "out_event_s10": 0.25,
                        "out_event_s8": 0.25,
                        }

    else:
        x_sw = Conv1D(32, kernel_size=3, strides=1, activation="relu", padding="same")(x)
        # x_sw = Conv1D(32, kernel_size=3, strides=1, activation="relu", padding="same")(x_sw)

        x_switch_s10p = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s10p", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
        x_switch_s10p = Activation("sigmoid", name="out_event_s10p")(x_switch_s10p)

        x_switch_s10 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s10", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
        x_switch_s10 = Activation("sigmoid", name="out_event_s10")(x_switch_s10)

        x_switch_s8 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s8", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
        x_switch_s8 = Activation("sigmoid", name="out_event_s8")(x_switch_s8)

        x_switch_s6 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s6", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
        x_switch_s6 = Activation("sigmoid", name="out_event_s6")(x_switch_s6)

        x_switch_s4 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s4", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
        x_switch_s4 = Activation("sigmoid", name="out_event_s4")(x_switch_s4)

        x_switch_s2 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s2", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
        x_switch_s2 = Activation("sigmoid", name="out_event_s2")(x_switch_s2)

        outputs = [x_sleepawake, x_switch_s10p, x_switch_s10, x_switch_s8, x_switch_s6, x_switch_s4, x_switch_s2, x_nan]
        loss = {"out_state": bce_loss_w_mask, # state_loss_focal,
            "out_state_nan": bce_loss_w_mask,
            "out_event_s10p": binary_focal_loss_switch,
            "out_event_s10": binary_focal_loss_switch,
            "out_event_s8": binary_focal_loss_switch,
            "out_event_s6": binary_focal_loss_switch,
            "out_event_s4": binary_focal_loss_switch,
            "out_event_s2": binary_focal_loss_switch,
            }
        loss_weights = {"out_state": 1.,
                        "out_state_nan": 1.,
                        "out_event_s10p": 0.25,
                        "out_event_s10": 0.35,
                        "out_event_s8": 0.35,
                        "out_event_s6": 0.25,
                        "out_event_s4": 0.25,
                        "out_event_s2": 0.25,
                        }

    """
    # HourGlass

    outputs_1st = outputs
    loss_1st = loss
    loss_weights_1st = loss_weights

        
    # すべての出力をstop_gradient
    second_inputs = [x_stem]# + outputs
    for out in outputs:
        second_inputs.append(Lambda(lambda x: tf.stop_gradient(x))(out))
    x = Concatenate()(second_inputs)
    shortcuts = []
    shortcuts.append(x)
    for i, ch in enumerate(block_channels[1:]):
        x = resblock_1d(x, ch, 3, 2, f"resblock_{i}_down1_2ndUnet")
        shortcuts.append(x)
    for i, ch in enumerate(upsample_blocks):
        x = UpSampling1D(size=2)(x)
        sc = shortcuts[-i-2]
        x = Concatenate()([x, sc])
        x = resblock_1d(x, ch, 3, 1, f"resblock_{i}_up1_2ndUnet")
        if outstride>1:
            if outstride!=4 and outstride!=2:
                raise NotImplementedError
            else:
                if i == len(upsample_blocks)-3 and outstride==4:
                    break
                elif i == len(upsample_blocks)-2 and outstride==2:
                    break
    
    x_s = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x_sleepawake = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_2ndUnet")(x_s)
    x_sleepawake = Activation("sigmoid", name="out_state_2ndUnet")(x_sleepawake)

    x_n = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x_nan = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_nan_2ndUnet")(x_n)
    x_nan = Activation("sigmoid", name="out_state_nan_2ndUnet")(x_nan)

    x_sw = Conv1D(32, kernel_size=3, strides=1, activation="relu", padding="same")(x)

    x_switch_s10p = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s10p_2ndUnet", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
    x_switch_s10p = Activation("sigmoid", name="out_event_s10p_2ndUnet")(x_switch_s10p)

    x_switch_s10 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s10_2ndUnet", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
    x_switch_s10 = Activation("sigmoid", name="out_event_s10_2ndUnet")(x_switch_s10)

    x_switch_s8 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s8_2ndUnet", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
    x_switch_s8 = Activation("sigmoid", name="out_event_s8_2ndUnet")(x_switch_s8)

    x_switch_s6 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s6_2ndUnet", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
    x_switch_s6 = Activation("sigmoid", name="out_event_s6_2ndUnet")(x_switch_s6)

    x_switch_s4 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s4_2ndUnet", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
    x_switch_s4 = Activation("sigmoid", name="out_event_s4_2ndUnet")(x_switch_s4)

    x_switch_s2 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s2_2ndUnet", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
    x_switch_s2 = Activation("sigmoid", name="out_event_s2_2ndUnet")(x_switch_s2)

    outputs_2nd = [x_sleepawake, x_switch_s10p, x_switch_s10, x_switch_s8, x_switch_s6, x_switch_s4, x_switch_s2, x_nan]
    loss_2nd = {"out_state_2ndUnet": bce_loss_w_mask,
        "out_state_nan_2ndUnet": bce_loss_w_mask,
        "out_event_s10p_2ndUnet": binary_focal_loss_switch,
        "out_event_s10_2ndUnet": binary_focal_loss_switch,
        "out_event_s8_2ndUnet": binary_focal_loss_switch,
        "out_event_s6_2ndUnet": binary_focal_loss_switch,
        "out_event_s4_2ndUnet": binary_focal_loss_switch,
        "out_event_s2_2ndUnet": binary_focal_loss_switch,
        }
    loss_weights_2nd = {"out_state_2ndUnet": 1.,
                    "out_state_nan_2ndUnet": 1.,
                    "out_event_s10p_2ndUnet": 0.25,
                    "out_event_s10_2ndUnet": 0.35,
                    "out_event_s8_2ndUnet": 0.35,
                    "out_event_s6_2ndUnet": 0.25,
                    "out_event_s4_2ndUnet": 0.25,
                    "out_event_s2_2ndUnet": 0.25,
                    }
    outputs = outputs_1st + outputs_2nd
    loss = {**loss_1st, **loss_2nd}
    loss_weights = {**loss_weights_1st, **loss_weights_2nd}
    """



    def cls_to_expect(inputs):
        s10, s8 = inputs
        return (s10 * 1 + s8 * 0.5) / 1.6

    # hourglass
    # ensemble_outputs = []
    # for out_1, out_2 in zip(outputs_1st, outputs_2nd):
    #     out = Lambda(lambda x: (x[0] + x[1]) / 2)([out_1, out_2])
    #     ensemble_outputs.append(out)


    # もともと
    event_outputs = [x_switch_s10p, x_switch_s10, x_switch_s8, x_switch_s6, x_switch_s4, x_switch_s2]
    # hourglass
    # event_outputs = ensemble_outputs[1:-1]

    x_switch_expect = Lambda(lambda x: tf.concat(x, axis=-1), name="infer_out_event")(event_outputs)

    # もともと
    outputs_infer = [x_sleepawake, x_switch_expect, x_nan]
    # hourglass
    # outputs_infer = ensemble_outputs[0:1] + [x_switch_expect] + ensemble_outputs[-1:]


    model = Model(inputs, outputs)
    model_infer = Model(inputs, outputs_infer)
    
    metrics = None # {"out_state": [accuracy_w_mask]}

    return model, model_infer, loss, loss_weights, metrics



def build_1d_model_multiclass_W(input_shape=(7168, 4), num_cls=11, D=True, Ws=[1,24], SSLmodel=False, outstride=1):

    def diffabs_features(feature_1ch):
         # anglezのみdiffをとる
        x_diff_1 = Lambda(lambda x: tf.roll(x, shift=1, axis=1) - x)(feature_1ch)
        x_diff_2 = Lambda(lambda x: tf.roll(x, shift=-1, axis=1) - x)(feature_1ch)
        x_diff = Lambda(lambda x: tf.math.abs(x[0]) + tf.math.abs(x[1]))([x_diff_1, x_diff_2])
        # x = Concatenate()([x, x_diff])
        return x_diff

    def angle_features(feature_1ch):
         # anglezのみdiffをとる
        x_diff = Lambda(lambda x: tf.roll(x, shift=1, axis=1) - x)(feature_1ch)
        x_diff_log1p = Lambda(lambda x: tf.math.log1p(tf.math.abs(x)) / 5.)(x_diff)
        x_diff_100 = Lambda(lambda x: tf.clip_by_value(x * 100., -1, 1))(x_diff)
        x_zero = Lambda(lambda x: tf.cast(tf.math.abs(x) < 1e-5, tf.float32))(x_diff)
        x = Concatenate()([x_diff, x_diff_log1p, x_diff_100, x_zero])
        return x_diff
    
    def normalize(x):
        x = x - tf.reduce_mean(x, axis=1, keepdims=True)
        x = x / (tf.math.reduce_std(x, axis=1, keepdims=True)+1e-7)
        return x


    def positioanl_encode(day_feature, omegas=[1, 24]):
        """
        0-1(0-24) -> sin, cos
        """
        def positional_encode_layer(x, min_timescale=4, max_timescale=24):
            outputs = []
            for omega in omegas:#, 24*4]: #, 24*4, 24*4*4]:
                sin = tf.math.sin(x * 2.0 * math.pi * omega)
                cos = tf.math.cos(x * 2.0 * math.pi * omega)
                outputs.append(sin)
                # outputs.append(cos)
            outputs = tf.concat(outputs, axis=-1)
            return outputs
        day_feature = Lambda(positional_encode_layer)(day_feature)
        return day_feature

    def ssl_loss(y_true, y_pred):
        # y_true はとりあえずダミー。ほんとはスプリットしなくても合致サンプル組はあるはず…。
        # y_pred は [2xb, 256]なので、前半後半にわけて、前半と後半でcos類似度をとる
        y_pred_1, y_pred_2 = tf.split(y_pred, 2, axis=0)
        y_pred_1 = tf.math.l2_normalize(y_pred_1, axis=-1)[:, tf.newaxis, :]
        y_pred_2 = tf.math.l2_normalize(y_pred_2, axis=-1)[tf.newaxis, :, :]
        cos_sim_matrix = y_pred_1 * y_pred_2 # [b, b, 256]
        cos_sim_matrix = 10 * tf.reduce_sum(cos_sim_matrix, axis=-1) # [b, b] # 10 is temperature
        # sigmoid
        cos_sim_matrix = tf.math.sigmoid(cos_sim_matrix)

        y_true = tf.eye(tf.shape(cos_sim_matrix)[0], dtype=tf.float32)
        loss = bce_loss_w_mask(y_true, cos_sim_matrix)
        return loss

    block_channels = [32, 32, 64, 64, 128, 256, 512, 512, 512, 1024, 1024] + [1024]
    shortcuts = []

    inputs = Input(input_shape, name="inputs") # b, l, ch
    if SSLmodel:
        inputs_split = Lambda(lambda x: tf.concat(tf.split(x, 2, axis=1), axis=0), name="inputs_split")(inputs) # 2b, l/2, ch
    else:
        inputs_split = inputs

    # ['daily_step', 'anglez', 'anglez_simpleerror', 'anglez_simpleerror_span', 'anglez_nanexist', 'anglez_daystd', 'anglez_daymean', "anglez_daycounter", "anglez_nancounter", 'enmo']
    
    day_feature = Lambda(lambda x: x[...,0:1])(inputs_split)
    anglez = Lambda(lambda x: x[...,1:2])(inputs_split)
    enmo = Lambda(lambda x: x[...,9:10])(inputs_split)

    day_feature_pe = positioanl_encode(day_feature, omegas=Ws)
    anglez_dabs = diffabs_features(anglez)
    anglez_feat = angle_features(anglez)
    enmo_scaled = Lambda(lambda x: tf.clip_by_value(x*25, 0, 1))(enmo)
    enmo_under0001 = Lambda(lambda x: tf.cast(x<0.001, tf.float32))(enmo)
    
    stem_inputs_angle = Concatenate(axis=-1)([anglez, anglez_dabs, anglez_feat])
    stem_inputs_enmo = Concatenate(axis=-1)([enmo, enmo_scaled, enmo_under0001])
    stem_inputs_days = day_feature_pe
    stem_inputs_others = Lambda(lambda x: x[...,2:9])(inputs_split)


    # SplitStem
    stem_ch = 40

    def nan_model(inputs):
        stem_ch = 16
        block_channels = [32, 32, 64, 64, 128, 256, 512, 512, 512, 1024, 1024] + [1024]
        shortcuts = []
        x = inputs
        x = Conv1D(stem_ch, kernel_size=3, strides=1, padding="same", activation="relu", name="stem_conv_nan")(x)
        shortcuts.append(x)
        for i, ch in enumerate(block_channels[1:]):
            x = resblock_1d(x, ch, 3, 2, f"resblock_{i}_down1_nan")
            shortcuts.append(x)
        # upsample UNet
        upsample_blocks = block_channels[-2::-1]
        for i, ch in enumerate(upsample_blocks):
            x = UpSampling1D(size=2)(x)
            sc = shortcuts[-i-2]
            x = Concatenate()([x, sc])
            x = resblock_1d(x, ch, 3, 1, f"resblock_{i}_up1_nan")
        x_n = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(x)
        x_nan = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_nan")(x_n)
        x_nan = Activation("sigmoid", name="out_state_nan")(x_nan)

        return x_nan

    # nan_model_inputs = Concatenate(axis=-1)([stem_inputs_angle, stem_inputs_enmo])
    # x_nan = nan_model(nan_model_inputs)
    # x_nan_stopgrad = Lambda(lambda x: tf.stop_gradient(x))(x_nan)
    # stem_inputs_others = Concatenate(axis=-1)([stem_inputs_others, x_nan_stopgrad])


    stem_inputs_angle = Conv1D(stem_ch, kernel_size=3, strides=1, padding="same", activation="relu", name="stem_conv_angle")(stem_inputs_angle)
    stem_inputs_enmo = Conv1D(stem_ch, kernel_size=3, strides=1, padding="same", activation="relu", name="stem_conv_enmo")(stem_inputs_enmo)
    stem_inputs_others = Conv1D(stem_ch, kernel_size=3, strides=1, padding="same", activation="relu", name="stem_conv_others")(stem_inputs_others)
    # sum all stem outputs
    
    stem_inputs_days = Conv1D(stem_ch, kernel_size=3, strides=1, padding="same", activation="relu", name="stem_conv_days")(stem_inputs_days)
    x = [
        stem_inputs_angle,
        stem_inputs_enmo,
        stem_inputs_days,
        stem_inputs_others,
    ]
    x = Lambda(lambda x: x[0]/3 + x[1]/3 + x[2]/3 + x[3]/3, name="stem_inputs")(x)

    if D==True:
        x = Dropout(0.25, noise_shape=[1,1,None])(x) # input_shape[-1]+5

    shortcuts.append(x)
    for i, ch in enumerate(block_channels[1:]):
        x = resblock_1d(x, ch, 3, 2, f"resblock_{i}_down1")
        shortcuts.append(x)
    

    # upsample UNet
    upsample_blocks = block_channels[-2::-1]
    for i, ch in enumerate(upsample_blocks):
        x = UpSampling1D(size=2)(x)
        sc = shortcuts[-i-2]
        x = Concatenate()([x, sc])
        x = resblock_1d(x, ch, 3, 1, f"resblock_{i}_up1")
        if outstride>1:
            if outstride!=4 and outstride!=2:
                raise NotImplementedError
            else:
                if i == len(upsample_blocks)-3 and outstride==4:
                    break
                elif i == len(upsample_blocks)-2 and outstride==2:
                    break

    
    x_s = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    # x_s = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(x_s)
    x_sleepawake = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv")(x_s)
    x_sleepawake = Activation("sigmoid", name="out_state")(x_sleepawake)

    x_n = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    # x_n = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(x_n)
    x_nan = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_nan")(x_n)
    x_nan = Activation("sigmoid", name="out_state_nan")(x_nan)

    manyouts = True
    if not manyouts:
        x_sw = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(x)
        x_switch_s10 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s10", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
        x_switch_s10 = Activation("sigmoid", name="out_event_s10")(x_switch_s10)

        x_switch_s8 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s8", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
        x_switch_s8 = Activation("sigmoid", name="out_event_s8")(x_switch_s8)

        outputs = [x_sleepawake, x_switch_s10, x_switch_s8, x_nan]
        loss = {"out_state": bce_loss_w_mask,
            "out_state_nan": bce_loss_w_mask,
            "out_event_s10": binary_focal_loss_switch,
            "out_event_s8": binary_focal_loss_switch,
            }
        loss_weights = {"out_state": 1.,
                        "out_state_nan": 1.,
                        "out_event_s10": 0.25,
                        "out_event_s8": 0.25,
                        }

    else:
        x_sw = Conv1D(32, kernel_size=3, strides=1, activation="relu", padding="same")(x)
        # x_sw = Conv1D(32, kernel_size=3, strides=1, activation="relu", padding="same")(x_sw)

        x_switch_s10p = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s10p", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
        x_switch_s10p = Activation("sigmoid", name="out_event_s10p")(x_switch_s10p)

        x_switch_s10 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s10", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
        x_switch_s10 = Activation("sigmoid", name="out_event_s10")(x_switch_s10)

        x_switch_s8 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s8", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
        x_switch_s8 = Activation("sigmoid", name="out_event_s8")(x_switch_s8)

        x_switch_s6 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s6", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
        x_switch_s6 = Activation("sigmoid", name="out_event_s6")(x_switch_s6)

        x_switch_s4 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s4", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
        x_switch_s4 = Activation("sigmoid", name="out_event_s4")(x_switch_s4)

        x_switch_s2 = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s2", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw)
        x_switch_s2 = Activation("sigmoid", name="out_event_s2")(x_switch_s2)

        outputs = [x_sleepawake, x_switch_s10p, x_switch_s10, x_switch_s8, x_switch_s6, x_switch_s4, x_switch_s2, x_nan]
        loss = {"out_state": bce_loss_w_mask, # state_loss_focal,
            "out_state_nan": bce_loss_w_mask,
            "out_event_s10p": binary_focal_loss_switch,
            "out_event_s10": binary_focal_loss_switch,
            "out_event_s8": binary_focal_loss_switch,
            "out_event_s6": binary_focal_loss_switch,
            "out_event_s4": binary_focal_loss_switch,
            "out_event_s2": binary_focal_loss_switch,
            }
        loss_weights = {"out_state": 1.,
                        "out_state_nan": 1.,
                        "out_event_s10p": 0.25,
                        "out_event_s10": 0.35,
                        "out_event_s8": 0.35,
                        "out_event_s6": 0.25,
                        "out_event_s4": 0.25,
                        "out_event_s2": 0.25,
                        }



    def cls_to_expect(inputs):
        s10, s8 = inputs
        return (s10 * 1 + s8 * 0.5) / 1.6

    # hourglass
    # ensemble_outputs = []
    # for out_1, out_2 in zip(outputs_1st, outputs_2nd):
    #     out = Lambda(lambda x: (x[0] + x[1]) / 2)([out_1, out_2])
    #     ensemble_outputs.append(out)


    # もともと
    event_outputs = [x_switch_s10p, x_switch_s10, x_switch_s8, x_switch_s6, x_switch_s4, x_switch_s2]
    # hourglass
    # event_outputs = ensemble_outputs[1:-1]

    x_switch_expect = Lambda(lambda x: tf.concat(x, axis=-1), name="infer_out_event")(event_outputs)

    # もともと
    outputs_infer = [x_sleepawake, x_switch_expect, x_nan]
    # hourglass
    # outputs_infer = ensemble_outputs[0:1] + [x_switch_expect] + ensemble_outputs[-1:]


    model = Model(inputs, outputs)
    model_infer = Model(inputs, outputs_infer)
    
    metrics = None # {"out_state": [accuracy_w_mask]}

    return model, model_infer, loss, loss_weights, metrics


def build_1d_model_multiclass_twin(input_shape=(7168, 4), num_cls=11, D=True, Ws=[1,24], SSLmodel=False, outstride=1):

    def diffabs_features(feature_1ch):
         # anglezのみdiffをとる
        x_diff_1 = Lambda(lambda x: tf.roll(x, shift=1, axis=1) - x)(feature_1ch)
        x_diff_2 = Lambda(lambda x: tf.roll(x, shift=-1, axis=1) - x)(feature_1ch)
        x_diff = Lambda(lambda x: tf.math.abs(x[0]) + tf.math.abs(x[1]))([x_diff_1, x_diff_2])
        # x = Concatenate()([x, x_diff])
        return x_diff
    
    def normalize(x):
        x = x - tf.reduce_mean(x, axis=1, keepdims=True)
        x = x / (tf.math.reduce_std(x, axis=1, keepdims=True)+1e-7)
        return x


    def positioanl_encode(day_feature, omegas=[1, 24]):
        """
        0-1(0-24) -> sin, cos
        """
        def positional_encode_layer(x, min_timescale=4, max_timescale=24):
            outputs = []
            for omega in omegas:#, 24*4]: #, 24*4, 24*4*4]:
                sin = tf.math.sin(x * 2.0 * math.pi * omega)
                cos = tf.math.cos(x * 2.0 * math.pi * omega)
                outputs.append(sin)
                # outputs.append(cos)
            outputs = tf.concat(outputs, axis=-1)
            return outputs
        day_feature = Lambda(positional_encode_layer)(day_feature)
        return day_feature

    def ssl_loss(y_true, y_pred):
        # y_true はとりあえずダミー。ほんとはスプリットしなくても合致サンプル組はあるはず…。
        # y_pred は [2xb, 256]なので、前半後半にわけて、前半と後半でcos類似度をとる
        y_pred_1, y_pred_2 = tf.split(y_pred, 2, axis=0)
        y_pred_1 = tf.math.l2_normalize(y_pred_1, axis=-1)[:, tf.newaxis, :]
        y_pred_2 = tf.math.l2_normalize(y_pred_2, axis=-1)[tf.newaxis, :, :]
        cos_sim_matrix = y_pred_1 * y_pred_2 # [b, b, 256]
        cos_sim_matrix = 10 * tf.reduce_sum(cos_sim_matrix, axis=-1) # [b, b] # 10 is temperature
        # sigmoid
        cos_sim_matrix = tf.math.sigmoid(cos_sim_matrix)

        y_true = tf.eye(tf.shape(cos_sim_matrix)[0], dtype=tf.float32)
        loss = bce_loss_w_mask(y_true, cos_sim_matrix)
        return loss



    inputs = Input(input_shape, name="inputs") # b, l, ch
    anglez = Lambda(lambda x: x[...,1:2])(inputs)

    inputs_enmo = Lambda(lambda x: x[...,9:10])(inputs)
    inputs_anglez = Lambda(lambda x: x[...,:9])(inputs)

    x_dabs = diffabs_features(anglez)
    day_feature = Lambda(lambda x: x[...,0:1])(inputs_anglez)
    day_feature = positioanl_encode(day_feature, omegas=Ws)
    
    xa = Concatenate(axis=-1)([inputs_anglez, x_dabs])
    xa = Dropout(0.25, noise_shape=[1,1,None])(xa) # input_shape[-1]+5
    xa = Concatenate()([xa, day_feature])

    xe = inputs_enmo

    block_channels = [32, 32, 64, 64, 128, 256, 512, 512, 512, 1024, 1024]
    block_channels = block_channels
    shortcuts_A = []
    shortcuts_E = []

    xa = Conv1D(block_channels[0], kernel_size=3, strides=1, padding="same", activation="relu", name="stem_conv_a")(xa)
    xe = Conv1D(block_channels[0], kernel_size=3, strides=1, padding="same", activation="relu", name="stem_conv_e")(xe)
    shortcuts_A.append(xa)
    shortcuts_E.append(xe)

    for i, ch in enumerate(block_channels[1:]):
        xa = resblock_1d(xa, ch, 3, 2, f"resblock_{i}_down1a")
        xe = resblock_1d(xe, ch, 3, 2, f"resblock_{i}_down1e")
        shortcuts_A.append(xa)
        shortcuts_E.append(xe)
        
    xae_blend = Concatenate()([xa, xe])
    # upsample UNet
    upsample_blocks = block_channels[-2::-1]
    for i, ch in enumerate(upsample_blocks):
        xa = UpSampling1D(size=2)(xa)
        xe = UpSampling1D(size=2)(xe)
        xae_blend = UpSampling1D(size=2)(xae_blend)

        sca = shortcuts_A[-i-2]
        sce = shortcuts_E[-i-2]
        scae_blend = Concatenate()([sca, sce])
        xa = Concatenate()([xa, sca])
        xe = Concatenate()([xe, sce])
        xae_blend = Concatenate()([xae_blend, scae_blend])
        
        xa = resblock_1d(xa, ch, 3, 1, f"resblock_{i}_up1a")
        xe = resblock_1d(xe, ch, 3, 1, f"resblock_{i}_up1e")
        xae_blend = resblock_1d(xae_blend, ch, 3, 1, f"resblock_{i}_up1ae")

        if outstride>1:
            if outstride!=4 and outstride!=2:
                raise NotImplementedError
            else:
                if i == len(upsample_blocks)-3 and outstride==4:
                    break
                elif i == len(upsample_blocks)-2 and outstride==2:
                    break

    
    x_s_a = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(xa)
    x_sleepawake_a = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_a")(x_s_a)
    x_sleepawake_a = Activation("sigmoid", name="out_state_a")(x_sleepawake_a)

    x_n_a = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(xa)
    x_nan_a = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_nan_a")(x_n_a)
    x_nan_a = Activation("sigmoid", name="out_state_nan_a")(x_nan_a)


    x_s_e = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(xe)
    x_sleepawake_e = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_e")(x_s_e)
    x_sleepawake_e = Activation("sigmoid", name="out_state_e")(x_sleepawake_e)

    x_n_e = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(xe)
    x_nan_e = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_nan_e")(x_n_e)
    x_nan_e = Activation("sigmoid", name="out_state_nan_e")(x_nan_e)


    x_s_ae = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(xae_blend)
    x_sleepawake_ae = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_ae")(x_s_ae)
    x_sleepawake_ae = Activation("sigmoid", name="out_state_ae")(x_sleepawake_ae)

    x_n_ae = Conv1D(16, kernel_size=3, strides=1, activation="relu", padding="same")(xae_blend)
    x_nan_ae = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_nan_ae")(x_n_ae)
    x_nan_ae = Activation("sigmoid", name="out_state_nan_ae")(x_nan_ae)



    x_sw_a = Conv1D(32, kernel_size=3, strides=1, activation="relu", padding="same")(xa)
    x_switch_s10p_a = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s10p_a", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw_a)
    x_switch_s10p_a = Activation("sigmoid", name="out_event_s10p_a")(x_switch_s10p_a)
    x_switch_s10_a = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s10_a", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw_a)
    x_switch_s10_a = Activation("sigmoid", name="out_event_s10_a")(x_switch_s10_a)
    x_switch_s8_a = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s8_a", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw_a)
    x_switch_s8_a = Activation("sigmoid", name="out_event_s8_a")(x_switch_s8_a)
    x_switch_s6_a = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s6_a", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw_a)
    x_switch_s6_a = Activation("sigmoid", name="out_event_s6_a")(x_switch_s6_a)
    x_switch_s4_a = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s4_a", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw_a)
    x_switch_s4_a = Activation("sigmoid", name="out_event_s4_a")(x_switch_s4_a)
    x_switch_s2_a = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s2_a", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw_a)
    x_switch_s2_a = Activation("sigmoid", name="out_event_s2_a")(x_switch_s2_a)


    x_sw_e = Conv1D(32, kernel_size=3, strides=1, activation="relu", padding="same")(xe)
    x_switch_s10p_e = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s10p_e", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw_e)
    x_switch_s10p_e = Activation("sigmoid", name="out_event_s10p_e")(x_switch_s10p_e)
    x_switch_s10_e = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s10_e", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw_e)
    x_switch_s10_e = Activation("sigmoid", name="out_event_s10_e")(x_switch_s10_e)
    x_switch_s8_e = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s8_e", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw_e)
    x_switch_s8_e = Activation("sigmoid", name="out_event_s8_e")(x_switch_s8_e)
    x_switch_s6_e = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s6_e", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw_e)
    x_switch_s6_e = Activation("sigmoid", name="out_event_s6_e")(x_switch_s6_e)
    x_switch_s4_e = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s4_e", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw_e)
    x_switch_s4_e = Activation("sigmoid", name="out_event_s4_e")(x_switch_s4_e)
    x_switch_s2_e = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s2_e", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw_e)
    x_switch_s2_e = Activation("sigmoid", name="out_event_s2_e")(x_switch_s2_e)


    x_sw_ae = Conv1D(32, kernel_size=3, strides=1, activation="relu", padding="same")(xae_blend)
    x_switch_s10p_ae = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s10p_ae", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw_ae)
    x_switch_s10p_ae = Activation("sigmoid", name="out_event_s10p_ae")(x_switch_s10p_ae)
    x_switch_s10_ae = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s10_ae", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw_ae)
    x_switch_s10_ae = Activation("sigmoid", name="out_event_s10_ae")(x_switch_s10_ae)
    x_switch_s8_ae = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s8_ae", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw_ae)
    x_switch_s8_ae = Activation("sigmoid", name="out_event_s8_ae")(x_switch_s8_ae)
    x_switch_s6_ae = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s6_ae", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw_ae)
    x_switch_s6_ae = Activation("sigmoid", name="out_event_s6_ae")(x_switch_s6_ae)
    x_switch_s4_ae = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s4_ae", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw_ae)
    x_switch_s4_ae = Activation("sigmoid", name="out_event_s4_ae")(x_switch_s4_ae)
    x_switch_s2_ae = Conv1D(1, kernel_size=3, strides=1, padding="same", name="out_conv_switch_s2_ae", bias_initializer=tf.constant_initializer(-np.log((1.00-0.01)/0.01)))(x_sw_ae)
    x_switch_s2_ae = Activation("sigmoid", name="out_event_s2_ae")(x_switch_s2_ae)
    

    outputs = [x_sleepawake_a, x_switch_s10p_a, x_switch_s10_a, x_switch_s8_a, x_switch_s6_a, x_switch_s4_a, x_switch_s2_a, x_nan_a]
    outputs += [x_sleepawake_e, x_switch_s10p_e, x_switch_s10_e, x_switch_s8_e, x_switch_s6_e, x_switch_s4_e, x_switch_s2_e, x_nan_e]
    outputs += [x_sleepawake_ae, x_switch_s10p_ae, x_switch_s10_ae, x_switch_s8_ae, x_switch_s6_ae, x_switch_s4_ae, x_switch_s2_ae, x_nan_ae]

    loss = {"out_state_a": bce_loss_w_mask, "out_state_e": bce_loss_w_mask, "out_state_ae": bce_loss_w_mask, 
    "out_state_nan_a": bce_loss_w_mask, "out_state_nan_e": bce_loss_w_mask, "out_state_nan_ae": bce_loss_w_mask, 
    "out_event_s10p_a": binary_focal_loss_switch, "out_event_s10p_e": binary_focal_loss_switch, "out_event_s10p_ae": binary_focal_loss_switch,
    "out_event_s10_a": binary_focal_loss_switch, "out_event_s10_e": binary_focal_loss_switch, "out_event_s10_ae": binary_focal_loss_switch,
    "out_event_s8_a": binary_focal_loss_switch, "out_event_s8_e": binary_focal_loss_switch, "out_event_s8_ae": binary_focal_loss_switch,
    "out_event_s6_a": binary_focal_loss_switch, "out_event_s6_e": binary_focal_loss_switch, "out_event_s6_ae": binary_focal_loss_switch,
    "out_event_s4_a": binary_focal_loss_switch, "out_event_s4_e": binary_focal_loss_switch, "out_event_s4_ae": binary_focal_loss_switch,
    "out_event_s2_a": binary_focal_loss_switch, "out_event_s2_e": binary_focal_loss_switch, "out_event_s2_ae": binary_focal_loss_switch,
    }
    loss_weights = {"out_state_a": 1., "out_state_e": 1., "out_state_ae": 1.,
    "out_state_nan_a": 1., "out_state_nan_e": 1., "out_state_nan_ae": 1.,
    "out_event_s10p_a": 0.25, "out_event_s10p_e": 0.25, "out_event_s10p_ae": 0.25,
    "out_event_s10_a": 0.35, "out_event_s10_e": 0.35, "out_event_s10_ae": 0.35,
    "out_event_s8_a": 0.35, "out_event_s8_e": 0.35, "out_event_s8_ae": 0.35,
    "out_event_s6_a": 0.25, "out_event_s6_e": 0.25, "out_event_s6_ae": 0.25,
    "out_event_s4_a": 0.25, "out_event_s4_e": 0.25, "out_event_s4_ae": 0.25,
    "out_event_s2_a": 0.25, "out_event_s2_e": 0.25, "out_event_s2_ae": 0.25,
    }


    def cls_to_expect(inputs):
        s10, s8 = inputs
        return (s10 * 1 + s8 * 0.5) / 1.6

    # x_switch_expect = Lambda(cls_to_expect, name="infer_out_event")([x_switch_s10, x_switch_s8])
    x_switch_expect = Lambda(lambda x: tf.concat(x, axis=-1), name="infer_out_event")([x_switch_s10p_ae, x_switch_s10_ae, x_switch_s8_ae, x_switch_s6_ae, x_switch_s4_ae, x_switch_s2_ae])
    
    outputs_infer = [x_sleepawake_ae, x_switch_expect, x_nan_ae]

    model = Model(inputs, outputs)
    model_infer = Model(inputs, outputs_infer)
    
    metrics = {"out_state_ae": [accuracy_w_mask]}

    return model, model_infer, loss, loss_weights, metrics

def accuracy_w_mask(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    
    mask = tf.cast(y_true>=-1e-7, tf.float32)
    # minus value is invalid label.
    y_true = y_true # * mask
    y_pred = y_pred # * mask
    
    y_pred = tf.cast(y_pred>0.5, y_pred.dtype)
    y_true = tf.cast(y_true>0.5, y_true.dtype)
    return tf.reduce_sum(tf.cast(y_true==y_pred, tf.float32) * mask) / (tf.reduce_sum(mask)+1e-7)

def bce_loss_w_mask(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    
    mask = tf.cast(y_true>=-1e-7, tf.float32)
    # minus value is invalid label.
    # mask = tf.cast(y_true>=-1e-7, tf.float32)
    y_true = y_true * mask
    y_pred = y_pred * mask
    
    epsilon = K.epsilon()    
    y_true = tf.clip_by_value(y_true, epsilon, 1. - epsilon)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    loss = (- y_true * tf.math.log(y_pred) - (1.0-y_true) * tf.math.log(1.0-y_pred)) * mask
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask)+1e-7)

def bce_loss_w_mask_weighted(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_weight = tf.reshape(y_true[...,1], [-1])
    y_true = tf.reshape(y_true[...,0], [-1])
    y_pred = tf.reshape(y_pred, [-1])
    
    mask = tf.cast(y_true>=-1e-7, tf.float32)
    # minus value is invalid label.
    # mask = tf.cast(y_true>=-1e-7, tf.float32)
    y_true = y_true * mask
    y_pred = y_pred * mask
    
    epsilon = K.epsilon()    
    y_true = tf.clip_by_value(y_true, epsilon, 1. - epsilon)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    loss = (- y_true * tf.math.log(y_pred) - (1.0-y_true) * tf.math.log(1.0-y_pred)) * mask * y_weight
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask)+1e-7)


def bce_loss_w_mask_w_weight(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.reshape(y_true[...,0], [-1])
    y_weight = tf.reshape(y_true[...,1], [-1])
    y_pred = tf.reshape(y_pred, [-1])
    
    mask = tf.cast(y_true>=-1e-7, tf.float32)
    # minus value is invalid label.
    # mask = tf.cast(y_true>=-1e-7, tf.float32)
    y_true = y_true * mask
    y_pred = y_pred * mask
    
    epsilon = K.epsilon()    
    y_true = tf.clip_by_value(y_true, epsilon, 1. - epsilon)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    loss = (- y_true * tf.math.log(y_pred) - (1.0-y_true) * tf.math.log(1.0-y_pred)) * mask
    return tf.reduce_sum(loss * y_weight) / (tf.reduce_sum(y_weight * mask)+1e-7)

def state_loss_focal(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    mask = tf.cast(y_true>=-1e-7, tf.float32)
    # minus value is invalid label.
    # mask = tf.cast(y_true>=-1e-7, tf.float32)
    y_true = y_true * mask
    y_pred = y_pred * mask

    loss = binary_focal_loss(y_true, y_pred, gamma=2., alpha=.5) * mask
    return tf.reduce_mean(loss) * 20 # 適当。


def bce_loss_switch_w_mask(y_true, y_pred):
    """
    focal lossの方がいいかも
    """
    hardmode = True # nan部分は変化なしと予測する
    use_focal_loss = True
    neg_weight = 0.001
    y_true = tf.cast(y_true, y_pred.dtype)

    mask = tf.cast(y_true>=-1e-7, tf.float32)
    mask = mask[:,1:,:] * mask[:,:-1,:]
    y_true_switch = tf.math.abs(y_true[:,1:,:] - y_true[:,:-1,:])
    # 前後2ピクセル分も正解とする。maxpoolingで引き延ばす
    y_true_switch = tf.nn.max_pool1d(y_true_switch, 5, 1, padding="SAME")

    
    # 左端を調整。
    y_pred = y_pred[:, 1:, :]

    y_true_switch = tf.reshape(y_true_switch, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    mask = tf.reshape(mask, [-1])
    # mask = tf.cast(y_true>=-1e-7, tf.float32)
    # minus value is invalid label.
    # mask = tf.cast(y_true>=-1e-7, tf.float32)
    y_true_switch = y_true_switch * mask
    if use_focal_loss:
        loss = binary_focal_loss(y_true_switch, y_pred, gamma=2., alpha=.25)
    else:
        epsilon = K.epsilon()    
        y_true_switch = tf.clip_by_value(y_true_switch, epsilon, 1. - epsilon)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        loss = (- y_true_switch * tf.math.log(y_pred) - (1.0-y_true_switch) * tf.math.log(1.0-y_pred) * neg_weight)
    if not hardmode:
        loss = loss * mask
    
    return tf.reduce_sum(loss) / (tf.reduce_sum(y_true_switch)+1e-7) #(tf.reduce_sum(mask)+1e-7)

def binary_focal_loss_switch(y_true, y_pred):
    gamma=2.
    alpha=.25
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    loss = binary_focal_loss(y_true, y_pred, gamma=gamma, alpha=alpha)
    return tf.reduce_sum(loss) / (tf.reduce_sum(y_true)+1e-7)

def binary_focal_loss(y_true, y_pred, gamma=2., alpha=.25):
    """binary(?) focal loss 
    """
    epsilon = K.epsilon()

    pt_1 = tf.where(y_true>epsilon, y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(y_true<=epsilon, y_pred, tf.zeros_like(y_pred))
    
    pt_1 = tf.clip_by_value(pt_1, epsilon, 1. - epsilon)
    pt_0 = tf.clip_by_value(pt_0, epsilon, 1. - epsilon)
    loss = - (alpha * ((1.-pt_1)**gamma) * tf.math.log(pt_1)) *  y_true\
        - ((1-alpha) * (pt_0**gamma) * tf.math.log(1.-pt_0)) * (1.-y_true)
    return loss

def categorical_ce_loss(y_true, y_pred):
    """categorical focal loss 
    """
    num_pos = tf.reduce_sum(y_true[...,1:]) # negative label以外
    epsilon = K.epsilon()    
    y_true = tf.clip_by_value(y_true, epsilon, 1. - epsilon)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    loss = - y_true * tf.math.log(y_pred)

    return tf.reduce_sum(loss) / (num_pos+1e-7)

def categorical_log_loss(y_true, y_pred):
    """categorical focal loss 
    """
    num_ch = tf.shape(y_true)[-1]
    y_true = tf.reshape(y_true, [-1, num_ch])
    y_pred = tf.reshape(y_pred, [-1, num_ch])

    num_pos = tf.reduce_sum(y_true, axis=0) # negative label以外
    epsilon = K.epsilon()    
    y_true = tf.clip_by_value(y_true, epsilon, 1. - epsilon)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    loss = binary_focal_loss(y_true, y_pred, gamma=2., alpha=.25)
    loss = tf.reduce_sum(loss, axis=0) / (num_pos+1e-7)
    return tf.reduce_mean(loss)

def resblock(x, out_ch, kernel, stride, name, bias=True, use_se=True):
    inputs = x
    x = cbr(x, out_ch, kernel, stride, name+"_cbr0", bias)
    x = cbr(x, out_ch, kernel, 1, name+"_cbr1", bias)
    if use_se:
        x = se(x, out_ch, rate=4, name=name+"_se")
    #x_in = cbr(inputs, out_ch, 1, stride, name+"_cbr_shortcut", bias)
    x = Add()([x, inputs])
    return x

def cbr(x, out_layer, kernel, stride, name, bias=False, use_batchnorm=True):
    x = Conv2D(out_layer, kernel_size=kernel, strides=stride,use_bias=bias, padding="same", name=name+"_conv")(x)
    if use_batchnorm:
        x = batch_norm(name=name+"_bw")(x)
    else:
        raise Exception("tensorflow addons")
        #x = tfa.layers.GroupNormalization(name=name+"_bw")(x)
    x = Activation("relu",name=name+"_activation")(x)
    return x

def depth_cbr(x, kernel, stride, name, bias=True):#,use_bias=False
    x = DepthwiseConv2D(kernel_size=kernel, strides=stride,use_bias=bias,  padding="same", name=name+"_dwconv")(x)
    x = batch_norm(name=name+"_bw")(x)
    x = Activation("relu",name=name+"_activation")(x)
    return x

def cb(x, out_layer, kernel, stride, name, bias=True):
    x=Conv2D(out_layer, kernel_size=kernel, strides=stride,use_bias=bias,  padding="same", name=name+"_conv")(x)
    x = batch_norm(name=name+"_bw")(x)
    return x

def se(x_in, layer_n, rate, name):
    x = GlobalAveragePooling2D(name=name+"_squeeze")(x_in)
    x = Reshape((1,1,layer_n),name=name+"_reshape")(x)
    x = Conv2D(layer_n//rate, kernel_size=1,strides=1, name=name+"_reduce")(x)
    x= Activation("relu",name=name+"_relu")(x)
    x = Conv2D(layer_n, kernel_size=1,strides=1, name=name+"_expand")(x)
    x= Activation("sigmoid",name=name+"_sigmoid")(x)
    #x = Dense(layer_n, activation="sigmoid")(x)
    x_out=Multiply(name=name+"_excite")([x_in, x])
    return x_out

def aggregation_block(x_shallow, x_deep, deep_ch, out_ch):
    x_deep= Conv2DTranspose(deep_ch, kernel_size=2, strides=2, padding='same', use_bias=False)(x_deep)
    x_deep = batch_norm()(x_deep)   
    x_deep = LeakyReLU(alpha=0.1)(x_deep)
    x = Concatenate()([x_shallow, x_deep])
    x = Conv2D(out_ch, kernel_size=1, strides=1, padding="same")(x)
    x = batch_norm()(x)   
    x = LeakyReLU(alpha=0.1)(x)
    return x

def aggregation(skip_connections, output_layer_n, prefix=""):
    x_1= cbr(skip_connections["c1"], output_layer_n, 1, 1,prefix+"aggregation_1")
    x_1 = aggregation_block(x_1, skip_connections["c2"], output_layer_n, output_layer_n)
    x_2= cbr(skip_connections["c2"], output_layer_n, 1, 1,prefix+"aggregation_2")
    x_2 = aggregation_block(x_2, skip_connections["c3"], output_layer_n, output_layer_n)
    x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)
    x_3 = cbr(skip_connections["c3"], output_layer_n, 1, 1,prefix+"aggregation_3")
    x_3 = aggregation_block(x_3, skip_connections["c4"], output_layer_n, output_layer_n)
    x_2 = aggregation_block(x_2, x_3, output_layer_n, output_layer_n)
    x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)
    x_4 = cbr(skip_connections["c4"], output_layer_n, 1, 1,prefix+"aggregation_4")
    skip_connections_out=[x_1,x_2,x_3,x_4]
    return skip_connections_out

def aggregation_block_1d(x_shallow, x_deep, deep_ch, out_ch):
    x_deep = Conv1DTranspose(deep_ch, kernel_size=2, strides=2, padding='same', use_bias=False)(x_deep)
    x_deep = batch_norm()(x_deep)
    x_deep = LeakyReLU(alpha=0.1)(x_deep)
    x = Concatenate()([x_shallow, x_deep])
    x = Conv1D(out_ch, kernel_size=1, strides=1, padding="same")(x)
    x = batch_norm()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def _aggregation_1d(skip_connections, output_layer_n, prefix=""):
    """skip_connecionsは12個のリスト
    """
    x_1 = cbr_1d(skip_connections[0], output_layer_n, 1, 1, prefix + "aggregation_1")
    x_1 = aggregation_block_1d(x_1, skip_connections[1], output_layer_n, output_layer_n)
    x_2 = cbr_1d(skip_connections[1], output_layer_n, 1, 1, prefix + "aggregation_2")
    x_2 = aggregation_block_1d(x_2, skip_connections[2], output_layer_n, output_layer_n)
    x_1 = aggregation_block_1d(x_1, x_2, output_layer_n, output_layer_n)
    x_3 = cbr_1d(skip_connections[2], output_layer_n, 1, 1, prefix + "aggregation_3")
    x_3 = aggregation_block_1d(x_3, skip_connections[3], output_layer_n, output_layer_n)
    x_2 = aggregation_block_1d(x_2, x_3, output_layer_n, output_layer_n)
    x_1 = aggregation_block_1d(x_1, x_2, output_layer_n, output_layer_n)
    x_4 = cbr_1d(skip_connections[3], output_layer_n, 1, 1, prefix + "aggregation_4")
    x_4 = aggregation_block_1d(x_4, skip_connections[4], output_layer_n, output_layer_n)
    x_3 = aggregation_block_1d(x_3, x_4, output_layer_n, output_layer_n)
    x_2 = aggregation_block_1d(x_2, x_3, output_layer_n, output_layer_n)
    x_1 = aggregation_block_1d(x_1, x_2, output_layer_n, output_layer_n)
    x_5 = cbr_1d(skip_connections[4], output_layer_n, 1, 1, prefix + "aggregation_5")
    x_5 = aggregation_block_1d(x_5, skip_connections[5], output_layer_n, output_layer_n)
    x_4 = aggregation_block_1d(x_4, x_5, output_layer_n, output_layer_n)
    x_3 = aggregation_block_1d(x_3, x_4, output_layer_n, output_layer_n)
    x_2 = aggregation_block_1d(x_2, x_3, output_layer_n, output_layer_n)
    x_1 = aggregation_block_1d(x_1, x_2, output_layer_n, output_layer_n)
    x_6 = cbr_1d(skip_connections[5], output_layer_n, 1, 1, prefix + "aggregation_6")
    x_6 = aggregation_block_1d(x_6, skip_connections[6], output_layer_n, output_layer_n)
    x_5 = aggregation_block_1d(x_5, x_6, output_layer_n, output_layer_n)
    x_4 = aggregation_block_1d(x_4, x_5, output_layer_n, output_layer_n)
    x_3 = aggregation_block_1d(x_3, x_4, output_layer_n, output_layer_n)
    x_2 = aggregation_block_1d(x_2, x_3, output_layer_n, output_layer_n)
    x_1 = aggregation_block_1d(x_1, x_2, output_layer_n, output_layer_n)
    x_7 = cbr_1d(skip_connections[6], output_layer_n, 1, 1, prefix + "aggregation_7")
    x_7 = aggregation_block_1d(x_7, skip_connections[7], output_layer_n, output_layer_n)
    x_6 = aggregation_block_1d(x_6, x_7, output_layer_n, output_layer_n)
    x_5 = aggregation_block_1d(x_5, x_6, output_layer_n, output_layer_n)
    x_4 = aggregation_block_1d(x_4, x_5, output_layer_n, output_layer_n)
    x_3 = aggregation_block_1d(x_3, x_4, output_layer_n, output_layer_n)
    x_2 = aggregation_block_1d(x_2, x_3, output_layer_n, output_layer_n)
    x_1 = aggregation_block_1d(x_1, x_2, output_layer_n, output_layer_n)
    x_8 = cbr_1d(skip_connections[7], output_layer_n, 1, 1, prefix + "aggregation_8")
    x_8 = aggregation_block_1d(x_8, skip_connections[8], output_layer_n, output_layer_n)
    x_7 = aggregation_block_1d(x_7, x_8, output_layer_n, output_layer_n)
    x_6 = aggregation_block_1d(x_6, x_7, output_layer_n, output_layer_n)
    x_5 = aggregation_block_1d(x_5, x_6, output_layer_n, output_layer_n)
    x_4 = aggregation_block_1d(x_4, x_5, output_layer_n, output_layer_n)
    x_3 = aggregation_block_1d(x_3, x_4, output_layer_n, output_layer_n)
    x_2 = aggregation_block_1d(x_2, x_3, output_layer_n, output_layer_n)
    x_1 = aggregation_block_1d(x_1, x_2, output_layer_n, output_layer_n)
    x_9 = cbr_1d(skip_connections[8], output_layer_n, 1, 1, prefix + "aggregation_9")
    x_9 = aggregation_block_1d(x_9, skip_connections[9], output_layer_n, output_layer_n)
    x_8 = aggregation_block_1d(x_8, x_9, output_layer_n, output_layer_n)
    x_7 = aggregation_block_1d(x_7, x_8, output_layer_n, output_layer_n)
    x_6 = aggregation_block_1d(x_6, x_7, output_layer_n, output_layer_n)
    x_5 = aggregation_block_1d(x_5, x_6, output_layer_n, output_layer_n)
    x_4 = aggregation_block_1d(x_4, x_5, output_layer_n, output_layer_n)
    x_3 = aggregation_block_1d(x_3, x_4, output_layer_n, output_layer_n)
    x_2 = aggregation_block_1d(x_2, x_3, output_layer_n, output_layer_n)
    x_1 = aggregation_block_1d(x_1, x_2, output_layer_n, output_layer_n)
    x_10 = cbr_1d(skip_connections[9], output_layer_n, 1, 1, prefix + "aggregation_10")
    x_10 = aggregation_block_1d(x_10, skip_connections[10], output_layer_n, output_layer_n)
    x_9 = aggregation_block_1d(x_9, x_10, output_layer_n, output_layer_n)
    x_8 = aggregation_block_1d(x_8, x_9, output_layer_n, output_layer_n)
    x_7 = aggregation_block_1d(x_7, x_8, output_layer_n, output_layer_n)
    x_6 = aggregation_block_1d(x_6, x_7, output_layer_n, output_layer_n)
    x_5 = aggregation_block_1d(x_5, x_6, output_layer_n, output_layer_n)
    x_4 = aggregation_block_1d(x_4, x_5, output_layer_n, output_layer_n)
    x_3 = aggregation_block_1d(x_3, x_4, output_layer_n, output_layer_n)
    x_2 = aggregation_block_1d(x_2, x_3, output_layer_n, output_layer_n)
    x_1 = aggregation_block_1d(x_1, x_2, output_layer_n, output_layer_n)
    x_11 = cbr_1d(skip_connections[10], output_layer_n, 1, 1, prefix + "aggregation_11")
    x_11 = aggregation_block_1d(x_11, skip_connections[11], output_layer_n, output_layer_n)
    x_10 = aggregation_block_1d(x_10, x_11, output_layer_n, output_layer_n)
    x_9 = aggregation_block_1d(x_9, x_10, output_layer_n, output_layer_n)
    x_8 = aggregation_block_1d(x_8, x_9, output_layer_n, output_layer_n)
    x_7 = aggregation_block_1d(x_7, x_8, output_layer_n, output_layer_n)
    x_6 = aggregation_block_1d(x_6, x_7, output_layer_n, output_layer_n)
    x_5 = aggregation_block_1d(x_5, x_6, output_layer_n, output_layer_n)
    x_4 = aggregation_block_1d(x_4, x_5, output_layer_n, output_layer_n)
    x_3 = aggregation_block_1d(x_3, x_4, output_layer_n, output_layer_n)
    x_2 = aggregation_block_1d(x_2, x_3, output_layer_n, output_layer_n)
    x_1 = aggregation_block_1d(x_1, x_2, output_layer_n, output_layer_n)
    
    skip_connections_out = [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, skip_connections[11]]
    return skip_connections_out


def aggregation_1d(skip_connections, output_layer_n, prefix=""):
    """skip_connecionsは12個のリスト
    """
    x_1 = cbr_1d(skip_connections[0], output_layer_n, 1, 1, prefix + "aggregation_1")
    x_1 = aggregation_block_1d(x_1, skip_connections[1], output_layer_n, output_layer_n)
    x_2 = cbr_1d(skip_connections[1], output_layer_n*2, 1, 1, prefix + "aggregation_2")
    x_2 = aggregation_block_1d(x_2, skip_connections[2], output_layer_n*2, output_layer_n)
    x_1 = aggregation_block_1d(x_1, x_2, output_layer_n, output_layer_n)
    x_3 = cbr_1d(skip_connections[2], output_layer_n*2, 1, 1, prefix + "aggregation_3")
    x_3 = aggregation_block_1d(x_3, skip_connections[3], output_layer_n*2, output_layer_n*2)
    x_2 = aggregation_block_1d(x_2, x_3, output_layer_n*2, output_layer_n)
    x_1 = aggregation_block_1d(x_1, x_2, output_layer_n, output_layer_n)
    x_4 = cbr_1d(skip_connections[3], output_layer_n*4, 1, 1, prefix + "aggregation_4")
    x_4 = aggregation_block_1d(x_4, skip_connections[4], output_layer_n*4, output_layer_n*2)
    x_3 = aggregation_block_1d(x_3, x_4, output_layer_n*2, output_layer_n*2)
    x_2 = aggregation_block_1d(x_2, x_3, output_layer_n*2, output_layer_n)
    x_1 = aggregation_block_1d(x_1, x_2, output_layer_n, output_layer_n)
    x_5 = cbr_1d(skip_connections[4], output_layer_n*4, 1, 1, prefix + "aggregation_5")
    x_5 = aggregation_block_1d(x_5, skip_connections[5], output_layer_n*4, output_layer_n*4)
    x_4 = aggregation_block_1d(x_4, x_5, output_layer_n*4, output_layer_n*2)
    x_3 = aggregation_block_1d(x_3, x_4, output_layer_n*2, output_layer_n*2)
    x_2 = aggregation_block_1d(x_2, x_3, output_layer_n*2, output_layer_n)
    x_1 = aggregation_block_1d(x_1, x_2, output_layer_n, output_layer_n)
    x_6 = cbr_1d(skip_connections[5], output_layer_n*8, 1, 1, prefix + "aggregation_6")
    x_6 = aggregation_block_1d(x_6, skip_connections[6], output_layer_n*8, output_layer_n*4)
    x_5 = aggregation_block_1d(x_5, x_6, output_layer_n*4, output_layer_n*4)
    x_4 = aggregation_block_1d(x_4, x_5, output_layer_n*4, output_layer_n*2)
    x_3 = aggregation_block_1d(x_3, x_4, output_layer_n*2, output_layer_n*2)
    x_2 = aggregation_block_1d(x_2, x_3, output_layer_n*2, output_layer_n)
    x_1 = aggregation_block_1d(x_1, x_2, output_layer_n, output_layer_n)
    x_7 = cbr_1d(skip_connections[6], output_layer_n*8, 1, 1, prefix + "aggregation_7")
    x_7 = aggregation_block_1d(x_7, skip_connections[7], output_layer_n*8, output_layer_n*8)
    x_6 = aggregation_block_1d(x_6, x_7, output_layer_n*8, output_layer_n*4)
    x_5 = aggregation_block_1d(x_5, x_6, output_layer_n*4, output_layer_n*4)
    x_4 = aggregation_block_1d(x_4, x_5, output_layer_n*4, output_layer_n*2)
    x_3 = aggregation_block_1d(x_3, x_4, output_layer_n*2, output_layer_n*2)
    x_2 = aggregation_block_1d(x_2, x_3, output_layer_n*2, output_layer_n)
    x_1 = aggregation_block_1d(x_1, x_2, output_layer_n, output_layer_n)
    x_8 = cbr_1d(skip_connections[7], output_layer_n*16, 1, 1, prefix + "aggregation_8")
    x_8 = aggregation_block_1d(x_8, skip_connections[8], output_layer_n*16, output_layer_n*8)
    x_7 = aggregation_block_1d(x_7, x_8, output_layer_n*8, output_layer_n*8)
    x_6 = aggregation_block_1d(x_6, x_7, output_layer_n*8, output_layer_n*4)
    x_5 = aggregation_block_1d(x_5, x_6, output_layer_n*4, output_layer_n*4)
    x_4 = aggregation_block_1d(x_4, x_5, output_layer_n*4, output_layer_n*2)
    x_3 = aggregation_block_1d(x_3, x_4, output_layer_n*2, output_layer_n*2)
    x_2 = aggregation_block_1d(x_2, x_3, output_layer_n*2, output_layer_n)
    x_1 = aggregation_block_1d(x_1, x_2, output_layer_n, output_layer_n)
    x_9 = cbr_1d(skip_connections[8], output_layer_n*16, 1, 1, prefix + "aggregation_9")
    x_9 = aggregation_block_1d(x_9, skip_connections[9], output_layer_n*16, output_layer_n*16)
    x_8 = aggregation_block_1d(x_8, x_9, output_layer_n*16, output_layer_n*8)
    x_7 = aggregation_block_1d(x_7, x_8, output_layer_n*8, output_layer_n*8)
    x_6 = aggregation_block_1d(x_6, x_7, output_layer_n*8, output_layer_n*4)
    x_5 = aggregation_block_1d(x_5, x_6, output_layer_n*4, output_layer_n*4)
    x_4 = aggregation_block_1d(x_4, x_5, output_layer_n*4, output_layer_n*2)
    x_3 = aggregation_block_1d(x_3, x_4, output_layer_n*2, output_layer_n*2)
    x_2 = aggregation_block_1d(x_2, x_3, output_layer_n*2, output_layer_n)
    x_1 = aggregation_block_1d(x_1, x_2, output_layer_n, output_layer_n)
    x_10 = cbr_1d(skip_connections[9], output_layer_n*32, 1, 1, prefix + "aggregation_10")
    x_10 = aggregation_block_1d(x_10, skip_connections[10], output_layer_n*32, output_layer_n*16)
    x_9 = aggregation_block_1d(x_9, x_10, output_layer_n*16, output_layer_n*16)
    x_8 = aggregation_block_1d(x_8, x_9, output_layer_n*16, output_layer_n*8)
    x_7 = aggregation_block_1d(x_7, x_8, output_layer_n*8, output_layer_n*8)
    x_6 = aggregation_block_1d(x_6, x_7, output_layer_n*8, output_layer_n*4)
    x_5 = aggregation_block_1d(x_5, x_6, output_layer_n*4, output_layer_n*4)
    x_4 = aggregation_block_1d(x_4, x_5, output_layer_n*4, output_layer_n*2)
    x_3 = aggregation_block_1d(x_3, x_4, output_layer_n*2, output_layer_n*2)
    x_2 = aggregation_block_1d(x_2, x_3, output_layer_n*2, output_layer_n)
    x_1 = aggregation_block_1d(x_1, x_2, output_layer_n, output_layer_n)
    x_11 = cbr_1d(skip_connections[10], output_layer_n*16, 1, 1, prefix + "aggregation_11")
    x_11 = aggregation_block_1d(x_11, skip_connections[11], output_layer_n*16, output_layer_n*16)
    x_10 = aggregation_block_1d(x_10, x_11, output_layer_n*16, output_layer_n*16)
    x_9 = aggregation_block_1d(x_9, x_10, output_layer_n*16, output_layer_n*16)
    x_8 = aggregation_block_1d(x_8, x_9, output_layer_n*16, output_layer_n*8)
    x_7 = aggregation_block_1d(x_7, x_8, output_layer_n*8, output_layer_n*8)
    x_6 = aggregation_block_1d(x_6, x_7, output_layer_n*8, output_layer_n*4)
    x_5 = aggregation_block_1d(x_5, x_6, output_layer_n*4, output_layer_n*4)
    x_4 = aggregation_block_1d(x_4, x_5, output_layer_n*4, output_layer_n*2)
    x_3 = aggregation_block_1d(x_3, x_4, output_layer_n*2, output_layer_n*2)
    x_2 = aggregation_block_1d(x_2, x_3, output_layer_n*2, output_layer_n)
    x_1 = aggregation_block_1d(x_1, x_2, output_layer_n, output_layer_n)
    
    skip_connections_out = [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, skip_connections[11]]
    return skip_connections_out

def effv2_encoder(inputs, is_train, from_scratch, model_name="s"):
    skip_connections={}
    pretrained_model = effnetv2_model.get_model('efficientnetv2-{}'.format(model_name), 
                                                model_config={"bn_type":"tpu_bn" if USE_TPU else None},
                                                include_top=False, 
                                                pretrained=False,
                                                training=is_train,
                                                input_shape=(None,None,3),
                                                input_tensor=inputs,
                                                with_endpoints=True)
    if not from_scratch:
        pretrained_model.load_weights(WEIGHT_DIR + 'effv2-{}-21k.h5'.format(model_name), by_name=True, skip_mismatch=True)    

    skip_connections["c1"] = pretrained_model.output[1]
    skip_connections["c2"] = pretrained_model.output[2]
    skip_connections["c3"] = pretrained_model.output[3]
    skip_connections["c4"] = pretrained_model.output[4]
    x = pretrained_model.output[5]

    return x, skip_connections


def decoder(inputs, skip_connections, use_batchnorm=True, 
            num_channels = 32, minimum_stride=2, max_stride=128,
            prefix=""):
    if not minimum_stride in [1,2,4,8]:
        raise Exception("minimum stride must be 1 or 2 or 4 or 8")
    if not max_stride in [32,64,128]:
        raise Exception("maximum stride must be 32 or 64 or 128")
    outs = []
    skip_connections = aggregation(skip_connections, num_channels, prefix=prefix)
    
    x = Dropout(0.2,noise_shape=(None, 1, 1, 1),name=prefix+'top_drop')(inputs)
    
    if max_stride>32:#more_deep        
        x_64 = cbr(x, 256, 3, 2,prefix+"top_64", use_batchnorm=use_batchnorm)
        if max_stride>64:
            x_128 = cbr(x_64, 256, 3, 2,prefix+"top_128", use_batchnorm=use_batchnorm)
            outs.append(x_128)
            x_64u = UpSampling2D(size=(2, 2))(x_128)
            x_64 = Concatenate()([x_64, x_64u])
        x_64 = cbr(x_64, 256, 3, 1,prefix+"top_64u", use_batchnorm=use_batchnorm)
        outs.append(x_64)
        x_32u = UpSampling2D(size=(2, 2))(x_64)
        x = Concatenate()([x, x_32u])    
    #x = Lambda(add_coords)(x)    
    x = cbr(x, num_channels*16, 3, 1,prefix+"decode_1", use_batchnorm=use_batchnorm)
    outs.append(x)
    x = UpSampling2D(size=(2, 2))(x)#8->16 tconvのがいいか

    x = Concatenate()([x, skip_connections[3]])
    x = cbr(x, num_channels*8, 3, 1,prefix+"decode_2", use_batchnorm=use_batchnorm)
    outs.append(x)
    x = UpSampling2D(size=(2, 2))(x)#16->32
    
    x = Concatenate()([x, skip_connections[2]])
    x = cbr(x, num_channels*4, 3, 1,prefix+"decode_3", use_batchnorm=use_batchnorm)
    outs.append(x)
   
    if minimum_stride<=4:
        x = UpSampling2D(size=(2, 2))(x)#32->64 
        x = Concatenate()([x, skip_connections[1]])
        x = cbr(x, num_channels*2, 3, 1,prefix+"decode_4", use_batchnorm=use_batchnorm)
        outs.append(x)
    if minimum_stride<=2:    
        x = UpSampling2D(size=(2, 2))(x)#64->128
        x = Concatenate()([x, skip_connections[0]])
        x = cbr(x, num_channels, 3, 1,prefix+"decode_5", use_batchnorm=use_batchnorm)
        outs.append(x)
    if minimum_stride==1:
        x = UpSampling2D(size=(2, 2))(x)#128->256
        outs.append(x)
    return outs

def add_high_freq_coords(inputs):
    
    
    batch_num, height, width = tf.unstack(tf.shape(inputs))[:3]
    
    h_grid = tf.expand_dims(tf.linspace(0., 5.0, height), 1) % 1.
    h_grid = 4 * tf.maximum(h_grid, 1. - h_grid) - 3. # -1 ro 1
    h_grid = tf.tile(h_grid, [1, width])
    w_grid = tf.expand_dims(tf.linspace(0., 5.0, width), 0) % 1.
    w_grid = 4 * tf.maximum(w_grid, 1. - w_grid) - 3.
    w_grid = tf.tile(w_grid, [height,1])
    hw_grid = tf.concat([tf.expand_dims(h_grid, -1),tf.expand_dims(w_grid, -1)], axis=-1)
    hw_grid = tf.expand_dims(hw_grid, 0)
    hw_grid = tf.tile(hw_grid, [batch_num, 1, 1, 1])
    
    return tf.concat([inputs, hw_grid], axis=-1)

def add_coords(inputs):
    batch_num, height, width = tf.unstack(tf.shape(inputs))[:3]
    
    h_grid = tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1)
    h_grid = tf.tile(h_grid, [1, width])
    w_grid = tf.expand_dims(tf.linspace(-1.0, 1.0, width), 0)
    w_grid = tf.tile(w_grid, [height,1])
    hw_grid = tf.concat([tf.expand_dims(h_grid, -1),tf.expand_dims(w_grid, -1)], axis=-1)
    hw_grid = tf.expand_dims(hw_grid, 0)
    hw_grid = tf.tile(hw_grid, [batch_num, 1, 1, 1])
    return tf.concat([inputs, hw_grid], axis=-1)


def crop_resize_layer(inputs, crop_size=[16,16], num_ch=1, unbatch=True):
    images, boxes = inputs
    batch, num_box, _ = tf.unstack(tf.shape(boxes))
    boxes = tf.reshape(boxes, [-1, 4])
    
    box_indices = tf.tile(tf.reshape(tf.range(batch),[-1,1]),[1,num_box])
    box_indices = tf.reshape(box_indices, [batch*num_box])
    crop_images = tf.image.crop_and_resize(images, boxes, box_indices, crop_size, method='bilinear')
    if unbatch:
        crop_images = tf.reshape(crop_images, [batch*num_box, crop_size[0], crop_size[1], num_ch])
    else:
        crop_images = tf.reshape(crop_images, [batch, num_box, crop_size[0], crop_size[1], num_ch])

    return crop_images


def l2_regularization(y_true, y_pred):
    loss = tf.reduce_mean(y_pred**2)
    return loss

def bce_loss(y_true, y_pred):
    negative_weight = 0.5
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    
    # minus value is low weight label. (out of paper)
    weight = tf.cast(y_true>=-1e-7, tf.float32) * 0.95 + 0.05
    # minus value is invalid label.
    # mask = tf.cast(y_true>=-1e-7, tf.float32)
    y_true = y_true# * mask
    y_pred = y_pred# * mask
    
    epsilon = K.epsilon()    
    y_true = tf.clip_by_value(y_true, epsilon, 1. - epsilon)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    loss = - y_true * tf.math.log(y_pred) - (1.0-y_true) * tf.math.log(1.0-y_pred) * negative_weight
    return tf.reduce_mean(loss*weight)# / (tf.reduce_sum(weight)+1e-7)

def soft_bce_loss(ratio=0.05):
    def bce_loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        # minus value is low weight label. (out of paper)
        weight = tf.cast(y_true>=-1e-7, tf.float32) * 0.95 + 0.05
        # mask = tf.cast(y_true>=-1e-7, tf.float32)
        y_true = y_true# * mask
        y_pred = y_pred# * mask
    
        y_true = ((1.-y_true)*ratio + y_true*(1-ratio))
        
        epsilon = K.epsilon()    
        y_true = tf.clip_by_value(y_true, epsilon, 1. - epsilon)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        loss = - y_true * tf.math.log(y_pred) - (1.0-y_true) * tf.math.log(1.0-y_pred)
        return tf.reduce_mean(loss*weight)# / (tf.reduce_sum(mask)+1e-7) #sumにして全マスクで割るのもあり。
    return bce_loss

def matthews_correlation_fixed(y_true, y_pred, threshold=0.3):
    y_pred = tf.cast(y_pred>threshold, y_pred.dtype)
    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1.-y_pred))
    fp = tf.reduce_sum((1.-y_true) * y_pred)
    tn = tf.reduce_sum((1.-y_true) * (1.-y_pred))
    score = (tp*tn - fp*fn) / tf.math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)+1e-7)
    return score

"""
Competition Description:
    We evaluate how well your output image matches our reference image 
    using a modified version of the Sørensen–Dice coefficient, 
    where instead of using the F1 score, we are using the F0.5 score.
"""
def SorensenDice_wrapper(threshold=0.5, beta=0.5):
    def SorensenDice_coefficient(y_true, y_pred):
        if threshold > 0:
            y_pred = tf.cast(y_pred>threshold, y_pred.dtype)
        tp_area = tf.reduce_sum(y_true * y_pred)
        precision = tp_area / (tf.reduce_sum(y_pred)+1e-7)
        recall = tp_area / (tf.reduce_sum(y_true)+1e-7)
        fbeta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-7)
        return fbeta
    return SorensenDice_coefficient

def best_SorensenDice(y_true, y_pred):
    best_score = 0
    for threshold in np.arange(0.1, 0.9, 0.05):
        score = SorensenDice_wrapper(threshold=threshold, beta=0.5)(y_true, y_pred)
        best_score = tf.maximum(best_score, score)
    return best_score

def SorensenDice_loss(y_true, y_pred):
    beta = 0.5
    smooth_val = 50
    mask = tf.cast(y_true>=-1e-7, tf.float32)
    y_true = y_true * mask
    y_pred = y_pred * mask
    tp_area = tf.reduce_sum(y_true * y_pred)
    precision = tp_area / (tf.reduce_sum(y_pred)+1e-7)
    recall = tp_area / (tf.reduce_sum(y_true)+1e-7)
    fbeta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + smooth_val)
    return 1. - fbeta

def combined_loss(weight_bce=0.5, weight_sorensen=0.5):
    def loss(y_true, y_pred):
        return weight_bce * bce_loss(y_true, y_pred) + weight_sorensen * SorensenDice_loss(y_true, y_pred)
    return loss

def conv1d_encoder(inputs, num_layers, num_ch, kernel_size=3):
    """
    inputs: [batch, height, width, channel]
    returns: [batch, height, width, num_ch]
    """
    x = inputs
    # reshape [batch, height, width, channel] -> [batch, height, width, channel, 1]
    x = Lambda(lambda x: tf.expand_dims(x, -1))(x)
    for i in range(num_layers):
        x = Conv1D(num_ch, kernel_size, padding="valid", name=f"conv1d_{i}")(x)
        # x = BatchNormalization()(x)
        x = Activation("relu")(x)
    # average on channel axis
    x = Lambda(lambda x: tf.reduce_max(x, axis=-2))(x)
    return x

def dense_encoder(inputs, num_layers, num_ch):
    """
    inputs: [batch, height, width, channel]
    returns: [batch, height, width, num_ch]
    """
    x = inputs
    for i in range(num_layers):
        x = Dense(num_ch, name=f"dense_{i}")(x)
        #x = BatchNormalization()(x)
        x = Activation("relu")(x)
    # average on channel axis
    # x = Lambda(lambda x: tf.reduce_mean(x, axis=-2))(x)
    return x

def build_model_1d(input_shape=(256,256,65),
             backbone="effv2s", 
             minimum_stride=2, 
             max_stride = 64,
             is_train=True,
             from_scratch=False,
             ):
    """
    model inputs:
        - normalized rgb(d)
        - boxes(normalized coordinates to show box location. top,left,bottom,right)
    """
    input_rgb = Input(input_shape, name="input_rgb")#64,64,65
    # enc_inの真ん中のチャンネルを使う
    enc_in = Lambda(lambda x: x[:,:,:,10:-10])(input_rgb)
    # x = conv1d_encoder(enc_in, num_layers=5, num_ch=16)
    x = dense_encoder(enc_in, num_layers=3, num_ch=128)
    out_mask = Conv2D(1, activation="sigmoid", kernel_size=3, strides=1, 
                        padding="same", 
                        name="out_mask",)(x)
    
    inputs = [input_rgb]
    outputs = [out_mask]
    losses = {#"out_mask": SorensenDice_loss_wrapper,
              "out_mask": bce_loss,
              }
    loss_weights = {"out_mask": 1.,
                    #"out_mask": 1.,
                    }
    metrics = {"out_mask": ["accuracy", 
                            # SorensenDice_wrapper(threshold=0.5, beta=0.5), 
                            SorensenDice_wrapper(threshold=0.2, beta=0.5),
                            ]}
    model = Model(inputs, outputs)
    # print(model.summary())

    return model, losses, loss_weights, metrics
    
def build_model(input_shape=(256,256,3),
             backbone="effv2s", 
             minimum_stride=2, 
             max_stride = 64,
             is_train=True,
             from_scratch=False,
             ):
    """
    model inputs:
        - normalized rgb(d)
        - boxes(normalized coordinates to show box location. top,left,bottom,right)
    """
    input_rgb = Input(input_shape, name="input_rgb")#64,64,65
    enc_in = input_rgb
    # enc_inの真ん中のチャンネルを使う
    enc_in = Lambda(lambda x: x[:,:,:,9:-9])(enc_in) # デフォルト65-30=35 chがあるので、9:-9なら中央17ch (24:-24 if original data)
    # enc_in = Lambda(lambda x: tf.concat([x[:,:,:,9:17], x[:,:,:,17:25]], axis=0))(enc_in)
    x = dense_encoder(enc_in, num_layers=3, num_ch=32)
    # x = conv1d_encoder(enc_in, num_layers=3, num_ch=8, kernel_size=5)

    out_mask_dense = Conv2D(1, activation="sigmoid", kernel_size=3, strides=1, 
                            padding="same", 
                            name="out_mask_dense")(x)
    # out_mask_dense = Lambda(lambda x: tf.maximum(*tf.split(x, 2, axis=0)), name="out_mask_dense")(out_mask_dense)


    x = Conv2D(3, activation="sigmoid", kernel_size=1, strides=1, 
                 padding="same", 
                             name="conv_input")(x)
    # out_mask_denseを3chにtile
    # x = Lambda(lambda x: tf.tile(x, [1,1,1,3]), name="tiled_input")(out_mask_dense)


    model_names = {"effv2s":"s", "effv2m":"m", "effv2l":"l", "effv2xl":"xl"}
    if backbone not in model_names.keys():
        raise Exception("check backbone name")
    x, skip_connections = effv2_encoder(x, is_train, from_scratch, model_name = model_names[backbone])

    use_coord_conv = False

    if use_coord_conv:
        print("use coords")
        
        x = Lambda(add_coords, name="add_coords")(x)
    
    outs = decoder(x, skip_connections, use_batchnorm=True, 
                   num_channels=32, max_stride=max_stride, minimum_stride=minimum_stride)
    x = outs[-1]
    out_mask = Conv2D(1, activation="sigmoid", kernel_size=3, strides=1, 
                        padding="same", 
                        name="out_mask",)(x)
    # out_mask = Lambda(lambda x: tf.maximum(*tf.split(x, 2, axis=0)), name="out_mask")(out_mask)

    inputs = [input_rgb]
    #outputs = [out_mask]
    outputs = [out_mask_dense, out_mask]
    losses = {"out_mask_dense": bce_loss,
              "out_mask": bce_loss,
              # "out_mask": combined_loss(0.5, 0.5),
              }
    loss_weights = {"out_mask_dense": 0.05/10000,
                    "out_mask": 1.,
                    }
    metrics = {"out_mask": ["accuracy", 
                            # SorensenDice_wrapper(threshold=0.5, beta=0.5), 
                            SorensenDice_wrapper(threshold=0.2, beta=0.5), # さすがにバッチ平均とりにくいかも。
                            ],
                "out_mask_dense": [
                            SorensenDice_wrapper(threshold=0.2, beta=0.5),]}
    model = Model(inputs, outputs)


    return model, losses, loss_weights, metrics


if __name__ == "__main__":

    model = build_1d_model_multiclass(input_shape=(7168*2, 10))[0]
    print(model.summary())

    # test
    y_true = tf.ones((10, 1024, 1), dtype=tf.float32)
    inputs = tf.ones((10, 7168*2, 10), dtype=tf.float32)
    preds = model(inputs)
    for p in preds:
        print(p.shape)
    # bce_loss_switch_w_mask(y_true, y_pred)
    
    
    
