from __future__ import division

import os

import numpy as np
import tensorflow as tf
import random

import __init__

dtype = tf.float32 if __init__.config['dtype'] == 'float32' else tf.float64
minval = __init__.config['minval']
maxval = __init__.config['maxval']
mean = __init__.config['mean']
stddev = __init__.config['stddev']


def all_random_seed(random_seed=1):
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.compat.v1.set_random_seed(random_seed)



def get_loss(loss_func):
    loss_func = loss_func.lower()
    if loss_func == 'weight' or loss_func == 'weighted':
        return tf.nn.weighted_cross_entropy_with_logits
    elif loss_func == 'sigmoid':
        return tf.nn.sigmoid_cross_entropy_with_logits
    elif loss_func == 'softmax':
        return tf.nn.softmax_cross_entropy_with_logits

def get_optimizer(opt_algo):
    opt_algo = opt_algo.lower()
    if opt_algo == 'adaldeta':
        return tf.train.AdadeltaOptimizer
    elif opt_algo == 'adagrad':
        return tf.train.AdagradOptimizer
    elif opt_algo == 'adam':
        return tf.train.AdamOptimizer
    elif opt_algo == 'moment':
        return tf.train.MomentumOptimizer
    elif opt_algo == 'ftrl':
        return tf.train.FtrlOptimizer
    elif opt_algo == 'gd' or opt_algo == 'sgd':
        return tf.train.GradientDescentOptimizer
    elif opt_algo == 'padagrad':
        return tf.train.ProximalAdagradOptimizer
    elif opt_algo == 'pgd':
        return tf.train.ProximalGradientDescentOptimizer
    elif opt_algo == 'rmsprop':
        return tf.train.RMSPropOptimizer
    else:
        return tf.train.GradientDescentOptimizer



def get_variable(init_type='xavier', shape=None, name=None, minval=minval, maxval=maxval, mean=mean,
                 stddev=stddev, dtype=dtype):
    if type(init_type) is str:
        init_type = init_type.lower()

    if init_type == 'xavier':
        maxval = np.sqrt(6. / np.sum(shape))
        minval = -maxval
        return tf.get_variable(name=name, shape=shape, initializer=tf.random_uniform_initializer(minval=minval, maxval=maxval, dtype=dtype))
    elif init_type == 'zero' or init_type == '0' or init_type == 0:
        return tf.get_variable(name=name, shape=shape, initializer=tf.zeros_initializer(dtype=dtype))
    elif 'int' in init_type.__class__.__name__ or 'float' in init_type.__class__.__name__:
        return tf.get_variable(name=name, shape=shape, initializer=tf.random_uniform_initializer(minval=init_type,
                                                                                                 maxval=init_type, dtype=dtype))
    else:
        print("change init type:", init_type)
        exit(-1)


def linear(xw):
    with tf.name_scope('linear'):
        l = tf.squeeze(tf.reduce_sum(xw, 1))
    return l


def activate(weights, act_type):
    if type(act_type) is str:
        act_type = act_type.lower()
    if act_type == 'sigmoid':
        return tf.nn.sigmoid(weights)
    elif act_type == 'softmax':
        return tf.nn.softmax(weights)
    elif act_type == 'relu':
        return tf.nn.relu(weights)
    elif act_type == 'tanh':
        return tf.nn.tanh(weights)
    elif act_type == 'elu':
        return tf.nn.elu(weights)
    elif act_type == 'none':
        return weights
    else:
        return weights

def check(x):
    try:
        return x is not None and x is not False and float(x) > 0
    except TypeError:
        return True


def DNN(init, h, input_shape, hidden_unit, activation, keep_rate, name, training, use_bn=False, use_ln=False,
        initial_dropout=False):
    layer_kernels = []
    layer_biases = []
    for i in range(len(hidden_unit)):
        with tf.name_scope(name+'hidden_%d' % i):
            if initial_dropout:
                tmp_keep_rate = tf.where(training, keep_rate[0], np.ones_like(keep_rate[0]),
                                         name=name + 'keep_prob_init')
                h = tf.nn.dropout(h, rate=1 - tmp_keep_rate)
            tmp_keep_rate = tf.where(training, keep_rate[i], np.ones_like(keep_rate[i]), name=name+'keep_prob_%d'%i)
            wi = get_variable(init, name=name+'w_%d' % i, shape=[input_shape, hidden_unit[i]])
            bi = get_variable(0, name=name+'b_%d' % i, shape=[hidden_unit[i]])
            h = tf.matmul(h, wi) + bi
            if use_bn and i != len(hidden_unit)-1:
                h = tf.layers.batch_normalization(h, training=training, reuse=tf.AUTO_REUSE, name=name+'mlp_bn_%d'%i)
            if use_ln and i != len(hidden_unit)-1:
                h = tf.contrib.layers.layer_norm(h, reuse=tf.AUTO_REUSE, scope=name + 'mlp_ln_%d' % i)
            h = tf.nn.dropout(activate(h, activation[i]), rate=1-tmp_keep_rate)
            input_shape = hidden_unit[i]
            layer_kernels.append(wi)
            layer_biases.append(bi)
            print("hhh h shape:", h.shape)
    return h, layer_kernels, layer_biases


def DNN_multi_input_unshare(init, input_matrix0, input_matrix1, input_matrix2, input_shape, hidden_unit, activation,
                            keep_rate, name, training, use_bn=False, use_ln=False, reuse=False,
                            special_dropout=False, initial_dropout=True):
    layer_kernels = []
    layer_biases = []

    # current_shape = input_shape
    current_shape = input_matrix0.get_shape().as_list()[1]
    with tf.variable_scope("DNN_multi_input_unshare_"+name, reuse=reuse):

        h = input_matrix0
        if initial_dropout:
            tmp_keep_rate = tf.where(training, keep_rate[0], np.ones_like(keep_rate[0]), name=name + 'keep_prob_init')
            h = tf.nn.dropout(h, rate=1 - tmp_keep_rate)
        for i in range(len(hidden_unit)):
            with tf.name_scope(name+'hidden_%d' % i):
                if keep_rate[i] != 1:
                    tmp_keep_rate = tf.where(training, keep_rate[i], np.ones_like(keep_rate[i]), name=name+'keep_prob_%d'%i)
                wi = get_variable(init, name=name+'w_%d' % i, shape=[current_shape, hidden_unit[i]])
                bi = get_variable(0, name=name+'b_%d' % i, shape=[hidden_unit[i]])
                h = tf.matmul(h, wi) + bi
                if i == 0:
                    h = tf.concat([h, input_matrix1], axis=-1)
                if i == 1:
                    h = tf.concat([h, input_matrix2], axis=-1)
                if use_ln and i != len(hidden_unit) - 1:
                    h = tf.contrib.layers.layer_norm(h, reuse=reuse, scope=name + 'mlp_ln_%d' % i)
                if use_bn and i != len(hidden_unit) - 1:
                    h = tf.layers.batch_normalization(h, training=training, reuse=reuse, name=name + 'mlp_bn_%d' % i)
                h = activate(h, activation[i])
                if special_dropout:
                    if i == 0 or i == 1:
                        h1 = h[:, :hidden_unit[i]]
                        h2 = h[:, hidden_unit[i]:]
                        if keep_rate[i] != 1:
                            h1 = tf.nn.dropout(h1, rate=1 - tmp_keep_rate)
                        print("test special dropout")
                        h = tf.concat([h1, h2], axis=-1)
                    else:
                        if keep_rate[i] != 1:
                            h = tf.nn.dropout(h, rate=1 - tmp_keep_rate)
                else:
                    if keep_rate[i] != 1:
                        h = tf.nn.dropout(h, rate=1-tmp_keep_rate)
                # current_shape = hidden_unit[i] + input_shape
                current_shape = h.get_shape().as_list()[1]
                layer_kernels.append(wi)
                layer_biases.append(bi)
                print("hhh h shape:", h.shape)
    return h, layer_kernels, layer_biases


def DNN_multi_input_unshare_darts(init, input_matrix0, input_matrix1, input_matrix2, input_shape, hidden_unit, activation,
                            keep_rate, name, training, use_bn=False, use_ln=False,
                                  other_input_seat=1):
    current_shape = input_shape
    layer_kernels = []
    layer_biases = []
    h = input_matrix0
    for i in range(len(hidden_unit)):
        with tf.name_scope(name + 'hidden_%d' % i):
            tmp_keep_rate = tf.where(training, keep_rate[i], np.ones_like(keep_rate[i]),
                                     name=name + 'keep_prob_%d' % i)
            wi = get_variable(init, name=name + 'w_%d' % i, shape=[current_shape, hidden_unit[i]])
            bi = get_variable(0, name=name + 'b_%d' % i, shape=[hidden_unit[i]])
            h = tf.matmul(h, wi) + bi
            if other_input_seat == 2:
                if i == 0:
                    h = tf.concat([h, input_matrix1], axis=-1)
                if i == 1:
                    h = tf.concat([h, input_matrix2], axis=-1)
            if use_ln and i != len(hidden_unit) - 1:
                h = tf.contrib.layers.layer_norm(h, reuse=tf.AUTO_REUSE, scope=name + 'mlp_ln_%d' % i)
            if use_bn and i != len(hidden_unit) - 1:
                h = tf.layers.batch_normalization(h, training=training, reuse=tf.AUTO_REUSE,
                                                  name=name + 'mlp_bn_%d' % i)
            h = tf.nn.dropout(activate(h, activation[i]), rate=1 - tmp_keep_rate)
            if other_input_seat == 1:
                if i == 0:
                    h = tf.concat([h, input_matrix1], axis=-1)
                if i == 1:
                    h = tf.concat([h, input_matrix2], axis=-1)
            current_shape = hidden_unit[i] + input_shape
            layer_kernels.append(wi)
            layer_biases.append(bi)
    return h, layer_kernels, layer_biases



def get_l2_loss(params, variables):
    _loss = None
    with tf.name_scope('l2_loss'):
        for p, v in zip(params, variables):
            print('add l2', p, v)
            if not type(p) is list:
                if check(p):
                    if type(v) is list:
                        for _v in v:
                            if _loss is None:
                                _loss = p * tf.nn.l2_loss(_v)
                            else:
                                _loss += p * tf.nn.l2_loss(_v)
                    else:
                        if _loss is None:
                            _loss = p * tf.nn.l2_loss(v)
                        else:
                            _loss += p * tf.nn.l2_loss(v)
            else:
                for _lp, _lv in zip(p, v):
                    if _loss is None:
                        _loss = _lp * tf.nn.l2_loss(_lv)
                    else:
                        _loss += _lp * tf.nn.l2_loss(_lv)
    return _loss


def get_l1_loss(params, variables):
    _loss = None
    with tf.name_scope('l1_loss'):
        for p, v in zip(params, variables):
            print('add l1', p, v)
            if not type(p) is list:
                if check(p):
                    if type(v) is list:
                        for _v in v:
                            if _loss is None:
                                _loss = p * tf.contrib.layers.l1_regularizer(0.5)(_v)
                            else:
                                _loss += p * tf.contrib.layers.l1_regularizer(0.5)(_v)
                    else:
                        if _loss is None:
                            _loss = p * tf.contrib.layers.l1_regularizer(0.5)(v)
                        else:
                            _loss += p * tf.contrib.layers.l1_regularizer(0.5)(v)
            else:
                for _lp, _lv in zip(p, v):
                    if _loss is None:
                        _loss = _lp * tf.contrib.layers.l1_regularizer(0.5)(_lv)
                    else:
                        _loss += _lp * tf.contrib.layers.l1_regularizer(0.5)(_lv)
    return _loss


def load_pb(pb):
    with tf.gfile.GFile(pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph