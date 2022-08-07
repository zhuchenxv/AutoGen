from __future__ import print_function

from abc import abstractmethod
from itertools import combinations
import numpy as np
import tensorflow as tf
# from tensorflow.python.framework import graph_util
# import keras.backend as K

import __init__
from tf_utils import *

dtype = __init__.config['dtype']

if dtype.lower() == 'float32' or dtype.lower() == 'float':
    dtype = tf.float32
elif dtype.lower() == 'float64':
    dtype = tf.float64



class Model:
    inputs = None
    outputs = None
    logits = None
    labels = None
    learning_rate = None
    loss = None
    l2_loss = None
    optimizer = None
    grad = None

    @abstractmethod
    def compile(self, **kwargs):
        pass

    def __str__(self):
        return self.__class__.__name__

class DSSM(Model):
    def __init__(self, init='xavier', user_field_num=None, item_field_num=None, item_multi_field_num=None,
                 user_feature_num=None, item_feature_num=None, embedding_size=None,
                 user_dnn_hidden_unit=None, user_dnn_activation=None, user_dnn_drop_rate=None,
                 item_dnn_hidden_unit=None, item_dnn_activation=None, item_dnn_drop_rate=None, similar_type="inner",
                 l2_v=0.0, l2_layer=0.0, user_bn=False, use_ln=False, initial_BN=False, initial_act=False,
                 initial_dropout=False,):
        self.l2_v = l2_v
        self.l2_layer = l2_layer

        with tf.name_scope('input'):
            self.user_input = tf.placeholder(tf.int32, [None, user_field_num], name='user_input')
            self.item_input = tf.placeholder(tf.int32, [None, item_field_num], name='item_input')
            if item_multi_field_num:
                self.item_multi_hot_input = tf.placeholder(tf.int32, [None, 3], name='item_multi_input')
                self.item_multi_hot_input_len = tf.placeholder(tf.float32, [None, 1], name='item_multi_input_len')
            self.labels = tf.placeholder(tf.float32, [None], name='label')
            self.training = tf.placeholder(dtype=tf.bool, name='training')
            self.all_ratings_user_mask_item_train = tf.placeholder(tf.float32, [None, None], name='all_ratings_user_mask_item_train')
            self.all_ratings_user_click_item_test = tf.placeholder(tf.float32, [None, None], name='all_ratings_user_click_item_test')

        with tf.name_scope('embedding'):
            user_v = get_variable(init_type=init, name='user_v', shape=[user_feature_num, embedding_size])
            item_v = get_variable(init_type=init, name='item_v', shape=[item_feature_num, embedding_size])
            user_xv = tf.gather(user_v, self.user_input)
            item_xv = tf.gather(item_v, self.item_input)
            if item_multi_field_num:
                item_multi_xv = tf.gather(item_v, self.item_multi_hot_input)
                item_multi_xv = tf.reduce_sum(item_multi_xv, axis=1) / self.item_multi_hot_input_len

        user_xv = tf.reshape(user_xv, (-1, user_field_num * embedding_size))
        item_xv = tf.reshape(item_xv, (-1, item_field_num * embedding_size))
        item_dnn_input = item_field_num * embedding_size
        if item_multi_field_num:
            item_xv = tf.concat([item_xv, item_multi_xv], axis=-1)
            item_dnn_input += embedding_size
        self.xv = tf.concat([user_xv, item_xv], axis=-1)
        user_dnn_out, user_kernel, user_bias = DNN(init, user_xv, embedding_size*user_field_num, user_dnn_hidden_unit,
                                                   user_dnn_activation, user_dnn_drop_rate, 'user_dnn_', self.training,
                                                   use_bn=user_bn, use_ln=use_ln, initial_dropout=initial_dropout)
        item_dnn_out, item_kernel, item_bias = DNN(init, item_xv, item_dnn_input, item_dnn_hidden_unit,
                                                   item_dnn_activation, item_dnn_drop_rate, 'item_dnn_', self.training,
                                                   use_bn=user_bn, use_ln=use_ln, initial_dropout=initial_dropout)
        print(user_dnn_out.shape)
        print(item_dnn_out.shape)
        self.layer_kernels = user_kernel+user_bias+item_kernel+item_bias
        if similar_type=="inner":
            score = tf.reduce_sum(tf.multiply(user_dnn_out, item_dnn_out), axis=-1)
            all_score = tf.matmul(user_dnn_out, item_dnn_out, transpose_b=True)
        elif similar_type=='cos':
            user_dnn_out_norm = tf.nn.l2_normalize(user_dnn_out, axis=-1)
            item_dnn_out_norm = tf.nn.l2_normalize(item_dnn_out, axis=-1)
            score = tf.reduce_sum(tf.multiply(user_dnn_out_norm, item_dnn_out_norm), axis=-1)
            all_score = tf.matmul(user_dnn_out_norm, item_dnn_out_norm, transpose_b=True)
        print(score.shape)
        print(all_score.shape)

        self.logits = score
        self.outputs = tf.nn.sigmoid(score)
        all_ratings = tf.nn.sigmoid(all_score)
        self.final_ratings = all_ratings

    def compile(self, loss=None, optimizer=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(loss(logits=self.logits, targets=self.labels, pos_weight=pos_weight))
                _loss_ = self.loss
                self.l2_loss = get_l2_loss([self.l2_v, self.l2_layer], [self.xv, self.layer_kernels])
                if self.l2_loss is not None:
                    _loss_ += self.l2_loss
                self.optimizer = optimizer.minimize(loss=_loss_, global_step=global_step)


class DSSM_multi_input(Model):
    def __init__(self, init='xavier', user_field_num=None, item_field_num=None, item_multi_field_num=None,
                 user_feature_num=None, item_feature_num=None, embedding_size=None,
                 user_dnn_hidden_unit=None, user_dnn_activation=None, user_dnn_drop_rate=None,
                 item_dnn_hidden_unit=None, item_dnn_activation=None, item_dnn_drop_rate=None, similar_type="inner",
                 l2_v=0.0, l2_layer=0.0, use_bn=False, use_ln=False, user_multi_layer=True, item_multi_layer=True,
                 initial_dropout=False, special_dropout=False):
        self.l2_v = l2_v
        self.l2_layer = l2_layer

        with tf.name_scope('input'):
            self.user_input = tf.placeholder(tf.int32, [None, user_field_num], name='user_input')
            self.item_input = tf.placeholder(tf.int32, [None, item_field_num], name='item_input')
            if item_multi_field_num:
                self.item_multi_hot_input = tf.placeholder(tf.int32, [None, 3], name='item_multi_input')
                self.item_multi_hot_input_len = tf.placeholder(tf.float32, [None, 1], name='item_multi_input_len')
            self.labels = tf.placeholder(tf.float32, [None], name='label')
            self.training = tf.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope('embedding'):
            user_v = get_variable(init_type=init, name='user_v', shape=[user_feature_num, embedding_size])
            item_v = get_variable(init_type=init, name='item_v', shape=[item_feature_num, embedding_size])
            user_xv = tf.gather(user_v, self.user_input)
            item_xv = tf.gather(item_v, self.item_input)
            if item_multi_field_num:
                item_multi_xv = tf.gather(item_v, self.item_multi_hot_input)
                item_multi_xv = tf.reduce_sum(item_multi_xv, axis=1) / self.item_multi_hot_input_len

        user_xv = tf.reshape(user_xv, (-1, user_field_num * embedding_size))
        item_xv = tf.reshape(item_xv, (-1, item_field_num * embedding_size))
        item_dnn_input = item_field_num * embedding_size
        if item_multi_field_num:
            item_xv = tf.concat([item_xv, item_multi_xv], axis=-1)
            item_dnn_input += embedding_size
        self.xv = tf.concat([user_xv, item_xv], axis=-1)
        if user_multi_layer:
            user_dnn_out, user_kernel, user_bias = DNN_multi_input_unshare(init, user_xv, user_xv, user_xv,
                                                                           embedding_size * user_field_num,
                                                                           user_dnn_hidden_unit, user_dnn_activation,
                                                                           user_dnn_drop_rate, 'user_dnn_', self.training,
                                                                           use_bn=use_bn, use_ln=use_ln,
                                                                           initial_dropout=initial_dropout,
                                                                           special_dropout=special_dropout)
        else:
            user_dnn_out, user_kernel, user_bias = DNN(init, user_xv, embedding_size * user_field_num,
                                                       user_dnn_hidden_unit,
                                                       user_dnn_activation, user_dnn_drop_rate, 'user_dnn_',
                                                       self.training,
                                                       use_bn=use_bn, use_ln=use_ln, initial_dropout=initial_dropout)
        if item_multi_layer:
            item_dnn_out, item_kernel, item_bias = DNN_multi_input_unshare(init, item_xv, item_xv, item_xv,
                                                                           item_dnn_input,
                                                                           item_dnn_hidden_unit,
                                                                           item_dnn_activation, item_dnn_drop_rate,
                                                                           'item_dnn_',
                                                                           self.training,
                                                                           use_bn=use_bn, use_ln=use_ln,
                                                                           initial_dropout=initial_dropout,
                                                                           special_dropout=special_dropout)
        else:
            item_dnn_out, item_kernel, item_bias = DNN(init, item_xv, item_dnn_input, item_dnn_hidden_unit,
                                                       item_dnn_activation, item_dnn_drop_rate, 'item_dnn_', self.training,
                                                       use_bn=use_bn, use_ln=use_ln, initial_dropout=initial_dropout)

        print(user_dnn_out.shape)
        print(item_dnn_out.shape)
        self.layer_kernels = user_kernel+user_bias+item_kernel+item_bias
        if similar_type=="inner":
            score = tf.reduce_sum(tf.multiply(user_dnn_out, item_dnn_out), axis=-1)
            all_score = tf.matmul(user_dnn_out, item_dnn_out, transpose_b=True)
        elif similar_type=='cos':
            user_dnn_out_norm = tf.nn.l2_normalize(user_dnn_out, axis=-1)
            item_dnn_out_norm = tf.nn.l2_normalize(item_dnn_out, axis=-1)
            score = tf.reduce_sum(tf.multiply(user_dnn_out_norm, item_dnn_out_norm), axis=-1)
            all_score = tf.matmul(user_dnn_out_norm, item_dnn_out_norm, transpose_b=True)

        self.logits = score
        self.outputs = tf.nn.sigmoid(score)
        all_ratings = tf.nn.sigmoid(all_score)
        self.final_ratings = all_ratings


    def compile(self, loss=None, optimizer=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(loss(logits=self.logits, targets=self.labels, pos_weight=pos_weight))
                _loss_ = self.loss
                self.l2_loss = get_l2_loss([self.l2_v, self.l2_layer], [self.xv, self.layer_kernels])
                if self.l2_loss is not None:
                    _loss_ += self.l2_loss
                self.optimizer = optimizer.minimize(loss=_loss_, global_step=global_step)



class DSSM_multi_input_darts(Model):
    def __init__(self, init='xavier', user_field_num=None, item_field_num=None, item_multi_field_num=None,
                 user_feature_num=None, item_feature_num=None, embedding_size=None,
                 user_dnn_hidden_unit=None, user_dnn_activation=None, user_dnn_drop_rate=None,
                 item_dnn_hidden_unit=None, item_dnn_activation=None, item_dnn_drop_rate=None, similar_type="inner",
                 l2_v=0.0, l2_layer=0.0, use_bn=False, use_ln=False, weight_base=0.6,
                 alpha_prune=True, beta_prune=True, l1_alpha=1e-4, l2_alpha=1e-4):
        self.l2_v = l2_v
        self.l2_layer = l2_layer
        self.l1_alpha = l1_alpha
        self.l2_alpha = l2_alpha
        self.alpha_prune = alpha_prune
        self.beta_prune = beta_prune

        with tf.name_scope('input'):
            self.user_input = tf.placeholder(tf.int32, [None, user_field_num], name='user_input')
            self.item_input = tf.placeholder(tf.int32, [None, item_field_num], name='item_input')
            if item_multi_field_num:
                self.item_multi_hot_input = tf.placeholder(tf.int32, [None, 3], name='item_multi_input')
                self.item_multi_hot_input_len = tf.placeholder(tf.float32, [None, 1], name='item_multi_input_len')
            self.labels = tf.placeholder(tf.float32, [None], name='label')
            self.training = tf.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope('embedding'):
            user_v = get_variable(init_type=init, name='user_v', shape=[user_feature_num, embedding_size])
            item_v = get_variable(init_type=init, name='item_v', shape=[item_feature_num, embedding_size])
            user_xv = tf.gather(user_v, self.user_input)
            item_xv = tf.gather(item_v, self.item_input)
            if item_multi_field_num:
                item_multi_xv = tf.gather(item_v, self.item_multi_hot_input)
                item_multi_xv = tf.reduce_sum(item_multi_xv, axis=1) / self.item_multi_hot_input_len

        user_xv = tf.reshape(user_xv, (-1, user_field_num * embedding_size))
        item_xv = tf.reshape(item_xv, (-1, item_field_num * embedding_size))
        item_dnn_input = item_field_num * embedding_size
        if item_multi_field_num:
            item_xv = tf.concat([item_xv, item_multi_xv], axis=-1)
            item_dnn_input += embedding_size
        self.xv = tf.concat([user_xv, item_xv], axis=-1)
        if self.alpha_prune:
            self.alpha_weight_user = tf.get_variable('alpha_weight_user', shape=[3, user_field_num, 1],
                                                     initializer=tf.random_uniform_initializer(
                                                         minval=weight_base - 0.001,
                                                         maxval=weight_base + 0.001))
        if self.beta_prune:
            self.beta_num = 8
            self.beta_weight_user = tf.get_variable('beta_weight_user', shape=[user_field_num, int(embedding_size/self.beta_num)],
                                                     initializer=tf.random_uniform_initializer(
                                                         minval=weight_base - 0.001,
                                                         maxval=weight_base + 0.001))

        # batch, 28, 40 第一维28上做bn
        user_xv_bn = tf.reshape(user_xv, (-1, user_field_num, embedding_size))
        user_xv_bn = tf.layers.batch_normalization(user_xv_bn, axis=1, training=self.training, scale=False,
                                                   center=False, name='dart_user_xv_bn1')  # batch, 28, 40
        if self.beta_prune:
            beta_weight_user1 = tf.expand_dims(self.beta_weight_user, axis=-1)
            user_xv_bn = tf.reshape(user_xv_bn, (-1, user_field_num, int(embedding_size/self.beta_num), self.beta_num))
            user_xv_bn = user_xv_bn * beta_weight_user1
            user_xv_bn = tf.reshape(user_xv_bn, (-1, user_field_num, embedding_size))

        if self.alpha_prune:
            user_matrix0 = user_xv_bn * self.alpha_weight_user[0]
            user_matrix1 = user_xv_bn * self.alpha_weight_user[1]
            user_matrix2 = user_xv_bn * self.alpha_weight_user[2]
        else:
            user_matrix0, user_matrix1, user_matrix2 = user_xv_bn, user_xv_bn, user_xv_bn
        user_matrix0 = tf.reshape(user_matrix0, (-1, user_field_num * embedding_size))
        user_matrix1 = tf.reshape(user_matrix1, (-1, user_field_num * embedding_size))
        user_matrix2 = tf.reshape(user_matrix2, (-1, user_field_num * embedding_size))

        item_matrix0, item_matrix1, item_matrix2 = item_xv, item_xv, item_xv

        user_dnn_out, user_kernel, user_bias = DNN_multi_input_unshare_darts(init, user_matrix0, user_matrix1,
                                                                             user_matrix2,
                                                                             embedding_size * user_field_num,
                                                                             user_dnn_hidden_unit, user_dnn_activation,
                                                                             user_dnn_drop_rate, 'user_dnn_',
                                                                             self.training,
                                                                             use_bn=use_bn, use_ln=use_ln,
                                                                             other_input_seat=1)
        item_dnn_out, item_kernel, item_bias = DNN_multi_input_unshare_darts(init, item_matrix0, item_matrix1,
                                                                             item_matrix2,
                                                                             item_dnn_input,
                                                                             item_dnn_hidden_unit,
                                                                             item_dnn_activation, item_dnn_drop_rate,
                                                                             'item_dnn_',
                                                                             self.training, use_bn=use_bn,
                                                                             use_ln=use_ln,
                                                                             other_input_seat=1)
        self.layer_kernels = user_kernel + user_bias + item_kernel + item_bias
        if similar_type == "inner":
            score = tf.reduce_sum(tf.multiply(user_dnn_out, item_dnn_out), axis=-1)
            all_score = tf.matmul(user_dnn_out, item_dnn_out, transpose_b=True)
        elif similar_type == 'cos':
            user_dnn_out_norm = tf.nn.l2_normalize(user_dnn_out, axis=-1)
            item_dnn_out_norm = tf.nn.l2_normalize(item_dnn_out, axis=-1)
            score = tf.reduce_sum(tf.multiply(user_dnn_out_norm, item_dnn_out_norm), axis=-1)
            all_score = tf.matmul(user_dnn_out_norm, item_dnn_out_norm, transpose_b=True)

        # with tf.name_scope('output'):
        self.logits = score
        self.outputs = tf.nn.sigmoid(score)
        all_ratings = tf.nn.sigmoid(all_score)
        self.final_ratings = all_ratings

    def analyse_structure(self, sess, print_full_weight=False, epoch=None):
        if self.alpha_prune:
            alpha_mask = sess.run(self.alpha_weight_user)  # 3, 28, 1
            alpha_mask = np.squeeze(alpha_mask, 2)
            alpha_mask = np.reshape(alpha_mask, (-1))
            print(alpha_mask.shape)
            if print_full_weight:
                outline = ""
                for j in range(alpha_mask.shape[0]):
                    outline += str(alpha_mask[j]) + ","
                outline += "\n"
                print("log avg auc all alpha user weights for(epoch:%s)" % (epoch), outline)
            print("alpha_mask user:", alpha_mask[:10])
        if self.beta_prune:
            beta_mask = sess.run(self.beta_weight_user)  # 28, 10
            beta_mask = np.reshape(beta_mask, (-1))
            print(beta_mask.shape)
            if print_full_weight:
                outline = ""
                for j in range(beta_mask.shape[0]):
                    outline += str(beta_mask[j]) + ","
                outline += "\n"
                print("log avg auc all beta user weights for(epoch:%s)" % (epoch), outline)
            print("beta_mask user", beta_mask[:10])

    def compile(self, loss=None, optimizer=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(loss(logits=self.logits, targets=self.labels, pos_weight=pos_weight))
                _loss_ = self.loss
                if self.alpha_prune and self.beta_prune:
                    l1_loss = get_l1_loss([self.l1_alpha, self.l1_alpha], [self.alpha_weight_user, self.beta_weight_user])
                    l2_loss = get_l2_loss([self.l2_alpha, self.l2_alpha], [self.alpha_weight_user, self.beta_weight_user])
                elif self.alpha_prune:
                    l1_loss = get_l1_loss([self.l1_alpha], [self.alpha_weight_user])
                    l2_loss = get_l2_loss([self.l2_alpha], [self.alpha_weight_user])
                else:
                    l1_loss = get_l1_loss([self.l1_alpha], [self.beta_weight_user])
                    l2_loss = get_l2_loss([self.l2_alpha], [self.beta_weight_user])
                if l1_loss is not None and l2_loss is not None:
                    self.l2_loss = l1_loss + l2_loss
                elif l1_loss is not None:
                    self.l2_loss = l1_loss
                else:
                    self.l2_loss = l2_loss
                # self.l2_loss = get_l2_loss([self.l2_v, self.l2_layer], [self.xv, self.layer_kernels])
                if self.l2_loss is not None:
                    _loss_ += self.l2_loss
                self.optimizer = optimizer.minimize(loss=_loss_, global_step=global_step)


class DSSM_multi_input_once_for_all(Model):
    def __init__(self, init='xavier', user_field_num=None, item_field_num=None, item_multi_field_num=None,
                 user_feature_num=None, item_feature_num=None, embedding_size=None,
                 user_dnn_hidden_unit=None, user_dnn_activation=None, user_dnn_drop_rate=None,
                 item_dnn_hidden_unit=None, item_dnn_activation=None, item_dnn_drop_rate=None, similar_type="inner",
                 l2_v=0.0, l2_layer=0.0, use_bn=False, use_ln=False, special_dropout=True, initial_dropout=False):
        self.l2_v = l2_v
        self.l2_layer = l2_layer

        with tf.name_scope('input'):
            self.user_input = tf.placeholder(tf.int32, [None, user_field_num], name='user_input')
            self.item_input = tf.placeholder(tf.int32, [None, item_field_num], name='item_input')
            if item_multi_field_num:
                self.item_multi_hot_input = tf.placeholder(tf.int32, [None, 3], name='item_multi_input')
                self.item_multi_hot_input_len = tf.placeholder(tf.float32, [None, 1], name='item_multi_input_len')
            self.labels = tf.placeholder(tf.float32, [None], name='label')
            self.soft_labels = tf.placeholder(tf.float32, [None], name='soft_label')
            self.training = tf.placeholder(dtype=tf.bool, name='training')

            self.all_ratings_user_mask_item_train = tf.placeholder(tf.float32, [None, None],
                                                                   name='all_ratings_user_mask_item_train')
            self.all_ratings_user_click_item_test = tf.placeholder(tf.float32, [None, None],
                                                                   name='all_ratings_user_click_item_test')

        with tf.name_scope('select_feature'):
            self.user_reserve_feature = tf.placeholder(tf.float32, [3, user_field_num], name='user_reserve_feature')
            self.beta_num = 8
            self.user_reserve_embedding = tf.placeholder(tf.float32, [user_field_num, int(embedding_size/self.beta_num)],
                                                         name='user_embedding_mask')

        with tf.name_scope('embedding'):
            user_v = get_variable(init_type=init, name='user_v', shape=[user_feature_num, embedding_size])
            item_v = get_variable(init_type=init, name='item_v', shape=[item_feature_num, embedding_size])
            user_xv = tf.gather(user_v, self.user_input)
            item_xv = tf.gather(item_v, self.item_input)
            if item_multi_field_num:
                item_multi_xv = tf.gather(item_v, self.item_multi_hot_input)
                item_multi_xv = tf.reduce_sum(item_multi_xv, axis=1) / self.item_multi_hot_input_len
                item_multi_xv = tf.expand_dims(item_multi_xv, axis=1)

        user_xv = tf.reshape(user_xv, (-1, user_field_num, int(embedding_size / self.beta_num), self.beta_num))
        user_embedding_mask1 = tf.expand_dims(self.user_reserve_embedding, axis=-1)
        user_xv *= user_embedding_mask1
        user_xv = tf.reshape(user_xv, (-1, user_field_num, embedding_size))

        if item_multi_field_num:
            item_xv = tf.concat([item_xv, item_multi_xv], axis=1)
        user_xv0 = tf.reshape(user_xv * tf.expand_dims(self.user_reserve_feature[0], axis=1), (-1, user_field_num * embedding_size))
        user_xv1 = tf.reshape(user_xv * tf.expand_dims(self.user_reserve_feature[1], axis=1), (-1, user_field_num * embedding_size))
        user_xv2 = tf.reshape(user_xv * tf.expand_dims(self.user_reserve_feature[2], axis=1), (-1, user_field_num * embedding_size))
        item_xv0 = tf.reshape(item_xv * 1, (-1, (item_field_num+1) * embedding_size))
        item_xv1 = tf.reshape(item_xv * 1, (-1, (item_field_num+1) * embedding_size))
        item_xv2 = tf.reshape(item_xv * 1, (-1, (item_field_num+1) * embedding_size))
        user_xv = tf.reshape(user_xv, (-1, user_field_num * embedding_size))
        item_xv = tf.reshape(item_xv, (-1, (item_field_num+1) * embedding_size))
        self.xv = tf.concat([user_xv, item_xv], axis=-1)
        user_dnn_out, user_kernel, user_bias = DNN_multi_input_unshare(init, user_xv0, user_xv1, user_xv2,
                                                               embedding_size * user_field_num,
                                                               user_dnn_hidden_unit, user_dnn_activation,
                                                               user_dnn_drop_rate, 'user_dnn_', self.training,
                                                               use_bn=use_bn, use_ln=use_ln, reuse=False,
                                                                       special_dropout=special_dropout,
                                                                       initial_dropout=initial_dropout)
        item_dnn_out, item_kernel, item_bias = DNN_multi_input_unshare(init, item_xv0, item_xv1, item_xv2,
                                                                       embedding_size * (item_field_num+1),
                                                               item_dnn_hidden_unit,
                                                               item_dnn_activation, item_dnn_drop_rate,
                                                               'item_dnn_',
                                                               self.training, use_bn=use_bn, use_ln=use_ln,
                                                                       reuse=False,
                                                                       special_dropout=special_dropout,
                                                                       initial_dropout=initial_dropout)
        self.layer_kernels = user_kernel+user_bias+item_kernel+item_bias
        score = tf.reduce_sum(tf.multiply(user_dnn_out, item_dnn_out), axis=-1)
        all_score = tf.matmul(user_dnn_out, item_dnn_out, transpose_b=True)
        self.logits = score
        self.outputs = tf.nn.sigmoid(score)
        all_ratings = tf.nn.sigmoid(all_score)
        self.final_ratings = all_ratings

        user_xv_new, item_xv_new = user_xv, item_xv
        user_xv_new0, user_xv_new1, user_xv_new2 = user_xv_new, user_xv_new, user_xv_new
        item_xv_new0, item_xv_new1, item_xv_new2 = item_xv_new, item_xv_new, item_xv_new
        user_dnn_out_new, _, _ = DNN_multi_input_unshare(init, user_xv_new0, user_xv_new1, user_xv_new2,
                                                         embedding_size * user_field_num,
                                                         user_dnn_hidden_unit, user_dnn_activation, user_dnn_drop_rate,
                                                         'user_dnn_', self.training,
                                                         use_bn=use_bn, use_ln=use_ln, reuse=True,
                                                         special_dropout=special_dropout,
                                                         initial_dropout=initial_dropout)
        item_dnn_out_new, _, _ = DNN_multi_input_unshare(init, item_xv_new0, item_xv_new1, item_xv_new2,
                                                         embedding_size * (item_field_num + 1),
                                                         item_dnn_hidden_unit, item_dnn_activation, item_dnn_drop_rate,
                                                         'item_dnn_',
                                                         self.training, use_bn=use_bn, use_ln=use_ln, reuse=True,
                                                         special_dropout=special_dropout,
                                                         initial_dropout=initial_dropout)
        score_new = tf.reduce_sum(tf.multiply(user_dnn_out_new, item_dnn_out_new), axis=-1)
        self.logits_new = score_new
        self.outputs_new = tf.nn.sigmoid(score_new)
        # not dropout output
        user_dnn_out_new_not_drop, _, _ = DNN_multi_input_unshare(init, user_xv_new0, user_xv_new1, user_xv_new2,
                                                         embedding_size * user_field_num,
                                                         user_dnn_hidden_unit, user_dnn_activation, [1.0, 1.0, 1.0],
                                                         'user_dnn_', False,
                                                         use_bn=use_bn, use_ln=use_ln, reuse=True,
                                                         special_dropout=special_dropout,
                                                         initial_dropout=initial_dropout)
        item_dnn_out_new_not_drop, _, _ = DNN_multi_input_unshare(init, item_xv_new0, item_xv_new1, item_xv_new2,
                                                         embedding_size * (item_field_num + 1),
                                                         item_dnn_hidden_unit, item_dnn_activation, [1.0, 1.0, 1.0],
                                                         'item_dnn_',
                                                         False, use_bn=use_bn, use_ln=use_ln, reuse=True,
                                                         special_dropout=special_dropout,
                                                         initial_dropout=initial_dropout)
        score_new_not_drop = tf.reduce_sum(tf.multiply(user_dnn_out_new_not_drop, item_dnn_out_new_not_drop), axis=-1)
        self.outputs_new_not_drop = tf.nn.sigmoid(score_new_not_drop)

    def compile(self, loss=None, optimizer=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                # mask feature  label
                loss1 = tf.reduce_mean(loss(logits=self.logits, targets=self.labels, pos_weight=pos_weight))
                # not mask feature label
                loss3 = tf.reduce_mean(loss(logits=self.logits_new, targets=self.labels, pos_weight=pos_weight))
                loss4 = tf.reduce_mean(
                    loss(logits=self.logits, targets=tf.stop_gradient(self.outputs_new_not_drop), pos_weight=pos_weight))
                _loss = loss1 + loss3 + loss4
                self.l2_loss = get_l2_loss([self.l2_v, self.l2_layer], [self.xv, self.layer_kernels])
                self.loss = _loss
                if self.l2_loss is not None:
                    _loss += self.l2_loss
                else:
                    self.l2_loss = tf.constant(0)
                self.optimizer = optimizer.minimize(loss=_loss, global_step=global_step)

