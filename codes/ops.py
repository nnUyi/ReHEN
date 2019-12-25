# -*- coding=utf-8 -*-

'''
    author: Youzhao Yang
    date: 12/25/2019
    github: https://github.com/nnuyi
'''

import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim

################ batch normalization setting ################
MOVING_AVERAGE_DECAY = 0.9997
BN_EPSILON = 0.001
BN_DECAY = MOVING_AVERAGE_DECAY
UPDATE_G_OPS_COLLECTION = 'Derain_g_update_ops'
UPDATE_D_OPS_COLLECTION = 'Derain_d_update_ops'
DERAIN_VARIABLES = 'Derain_variables'
#############################################################

# get variables for batch normalization
def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):

    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, DERAIN_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)
    
# batch normalization
def bn(x, c, update_ops_collection):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer)
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer)
    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer,
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(update_ops_collection, update_moving_mean)
    tf.add_to_collection(update_ops_collection, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        c, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    
    return x
