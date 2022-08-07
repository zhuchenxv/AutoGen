from __future__ import division
from __future__ import print_function

import datetime
import os
import time

import numpy as np
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
from sklearn.metrics import roc_auc_score, log_loss
from tf_utils import get_optimizer, get_loss
import heapq
import math
from test_matrix import test_one_user
import multiprocessing
import json

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0


class Trainer_eval:
    logdir = None
    session = None
    dataset = None
    model = None
    saver = None
    learning_rate = None
    train_pos_ratio = None
    test_pos_ratio = None
    ckpt_time = None

    def __init__(self, dataset=None, model=None, train_gen=None, test_gen=None, valid_gen=None,
                 opt='adam', epsilon=1e-8, initial_accumulator_value=1e-8, momentum=0.95,
                 loss='weighted', pos_weight=1.0,
                 n_epoch=1, train_per_epoch=10000, test_per_epoch=10000, early_stop_epoch=5,
                 batch_size=2000, learning_rate=1e-2, decay_rate=0.95,
                 test_every_epoch=1, logdir=None, pre_train_logdir=None):
        self.model = model
        self.dataset = dataset
        self.train_gen = train_gen
        self.test_gen = test_gen
        self.valid_gen = valid_gen
        optimizer = get_optimizer(opt)
        loss = get_loss(loss)
        self.pos_weight = pos_weight
        self.n_epoch = n_epoch
        self.train_per_epoch = train_per_epoch + 1
        self.early_stop_epoch = early_stop_epoch
        self.test_per_epoch = test_per_epoch
        self.batch_size = batch_size
        self._learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.test_every_epoch = test_every_epoch
        self.logdir = logdir
        self.max_result = None
        print("current max output:", 0)

        self.call_auc = roc_auc_score
        self.call_loss = log_loss
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        self.learning_rate = tf.placeholder("float")
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        tf.summary.scalar('global_step', self.global_step)
        print(optimizer)
        if opt == 'adam':
            opt = optimizer(learning_rate=self.learning_rate, epsilon=self.epsilon,
                            beta1=0.9, beta2=0.999)  # TODO fbh
        elif opt == 'adagrad':
            opt = optimizer(learning_rate=self.learning_rate, initial_accumulator_value=initial_accumulator_value)
        elif opt == 'moment':
            opt = optimizer(learning_rate=self.learning_rate, momentum=momentum)
        else:
            opt = optimizer(learning_rate=self.learning_rate, )  # TODO fbh
        self.model.compile(loss=loss, optimizer=opt, global_step=self.global_step, pos_weight=pos_weight)

        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())

        if pre_train_logdir is not None:
            module_file = tf.train.latest_checkpoint(pre_train_logdir)
            reader = pywrap_tensorflow.NewCheckpointReader(module_file)
            var_to_shape_map = reader.get_variable_to_shape_map()
            variables = tf.contrib.framework.get_variables_to_restore()
            variables_to_restore = []
            for v in variables:
                for key in var_to_shape_map:
                    if key == v.name[:-2]:
                        variables_to_restore.append(v)
                        break
            variables_not_restore = [v for v in variables if v not in variables_to_restore]
            print("load variable store/unstore number:", len(variables_to_restore), len(variables_not_restore))
            tf.train.Saver(variables_to_restore, max_to_keep=1).restore(self.session, module_file)

    def _run(self, fetches, feed_dict):
        return self.session.run(fetches=fetches, feed_dict=feed_dict)

    def test_all_user(self):
        Ks = [5, 10, 20, 50, 100]  # matrix@k
        result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}
        all_test_item_one_hot, all_test_item_multi_hot = self.dataset.get_all_item()
        for i in range(self.dataset.all_item_num):
            if all_test_item_one_hot[i][0] != i:
                print("wrong")
                exit(-1)
        batch_size = 1000
        part = 'new_test'  # test_old, validation, new_test
        test_data_param = {
            'part': part,
            'shuffle': False,
            'batch_size': batch_size,
        }
        count = 0
        pool = multiprocessing.Pool(4)
        if part == 'test_old':
            n_test_users = self.dataset.all_user_num
        elif part == 'validation':
            n_test_users = len(self.dataset.validation_user_order)
        else:
            n_test_users = len(self.dataset.only_test_user_order)
        print("begin test_users:", n_test_users)
        print("part:", part)
        tic = time.time()
        first_time = time.time()
        for batch_data in self.dataset.batch_generator(test_data_param):
            user_feature_batch, user_click_item_train_batch, user_click_item_test_batch = batch_data
            uid_batch = user_feature_batch[:, 0]
            current_batch_size = uid_batch.shape[0]
            feed_dict = {
                self.model.user_input: user_feature_batch,
                self.model.item_input: all_test_item_one_hot,
                self.model.item_multi_hot_input: all_test_item_multi_hot[:, :-1],
                self.model.item_multi_hot_input_len: all_test_item_multi_hot[:, -1:],
                self.model.training: False,
            }
            ratings_batch = self._run(fetches=self.model.final_ratings, feed_dict=feed_dict)

            user_batch_rating_uid = zip(ratings_batch, uid_batch, user_click_item_train_batch, user_click_item_test_batch)
            batch_result = pool.map(test_one_user, user_batch_rating_uid)
            count += len(batch_result)

            for re in batch_result:
                result['recall'] += re['recall'] / n_test_users
                result['ndcg'] += re['ndcg'] / n_test_users

            print("count:", count, time.time() - tic)
            tic = time.time()
        print("end:", time.time() - first_time)
        pool.close()
        print(result)
        # exit(-1)
        assert count == n_test_users
        return result

    def test_split_user(self,):
        Ks = [5, 10, 20, 50, 100]  # matrix@k
        result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}
        tic = time.time()
        all_test_item_one_hot, all_test_item_multi_hot = self.dataset.get_all_item()
        batch_size = 1000  # 1000
        n_test_users = len(self.dataset.only_test_user_order)
        count = 0
        pool = multiprocessing.Pool(4)

        split_result_all = []
        for split_user in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            test_data_param = {
                'part': 'new_test_split',  # new_test_way1_split
                'shuffle': False,
                'batch_size': batch_size,
                'split_user': split_user,
            }
            split_user_num = len(self.dataset.test_split_order[split_user])
            split_result = [0.0, 0.0]
            for batch_data in self.dataset.batch_generator(test_data_param):
                user_feature_batch, user_click_item_train_batch, user_click_item_test_batch = batch_data
                uid_batch = user_feature_batch[:, 0]
                current_batch_size = uid_batch.shape[0]
                feed_dict = {
                    self.model.user_input: user_feature_batch,
                    self.model.item_input: all_test_item_one_hot,
                    self.model.item_multi_hot_input: all_test_item_multi_hot[:, :-1],
                    self.model.item_multi_hot_input_len: all_test_item_multi_hot[:, -1:],
                    self.model.training: False,
                }
                ratings_batch = self._run(fetches=self.model.final_ratings, feed_dict=feed_dict)

                user_batch_rating_uid = zip(ratings_batch, uid_batch, user_click_item_train_batch,
                                            user_click_item_test_batch)
                batch_result = pool.map(test_one_user, user_batch_rating_uid)
                for re in batch_result:
                    result['recall'] += re['recall'] / n_test_users
                    result['ndcg'] += re['ndcg'] / n_test_users
                for re in batch_result:
                    split_result[0] += re['recall'][1] / split_user_num
                    split_result[1] += re['ndcg'][1] / split_user_num
                count += current_batch_size
            split_result = np.round(split_result, 4)
            print("part user:", split_user, "split_result:", split_result, "split_user_num:", split_user_num)
            split_result_all.append(split_result)
            # print("tmp count:", count)
        for i in range(len(split_result_all)):
            print(*split_result_all[i], sep=',')
        pool.close()
        assert count == n_test_users
        return result

    def test_all_user_one_tower(self):
        Ks = [5, 10, 20, 50, 100]  # matrix@k
        result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}
        all_test_item_one_hot, all_test_item_multi_hot = self.dataset.get_all_item()
        for i in range(self.dataset.all_item_num):
            if all_test_item_one_hot[i][0] != i:
                print("wrong")
                exit(-1)
        batch_size = 1000
        part = 'new_test'  # test_old, validation, new_test
        test_data_param = {
            'part': part,
            'shuffle': False,
            'batch_size': batch_size,
        }
        count = 0
        pool = multiprocessing.Pool(4)
        if part == 'test_old':
            n_test_users = self.dataset.all_user_num
        elif part == 'validation':
            n_test_users = len(self.dataset.validation_user_order)
        else:
            n_test_users = len(self.dataset.only_test_user_order)
        print("begin test_users:", n_test_users)
        print("part:", part)
        tic = time.time()
        first_time = time.time()
        for batch_data in self.dataset.batch_generator(test_data_param):
            user_feature_batch, user_click_item_train_batch, user_click_item_test_batch = batch_data
            uid_batch = user_feature_batch[:, 0]
            batch_user_num = user_feature_batch.shape[0]
            batch_item_num = all_test_item_one_hot.shape[0]

            user_feature_batch_input = np.tile(np.expand_dims(user_feature_batch, 1), [1, batch_item_num, 1])
            user_feature_batch_input = np.reshape(user_feature_batch_input, (-1, user_feature_batch.shape[1]))
            all_test_item_one_hot_input = np.tile(np.expand_dims(all_test_item_one_hot, 0), [batch_user_num, 1, 1])
            all_test_item_one_hot_input = np.reshape(all_test_item_one_hot_input, (-1, all_test_item_one_hot.shape[1]))
            all_test_item_multi_hot_input = np.tile(np.expand_dims(all_test_item_multi_hot, 0), [batch_user_num, 1, 1])
            all_test_item_multi_hot_input = np.reshape(all_test_item_multi_hot_input, (-1, all_test_item_multi_hot.shape[1]))
            feed_dict = {
                self.model.user_input: user_feature_batch_input,
                self.model.item_input: all_test_item_one_hot_input,
                self.model.item_multi_hot_input: all_test_item_multi_hot_input[:, :-1],
                self.model.item_multi_hot_input_len: all_test_item_multi_hot_input[:, -1:],
                self.model.training: False,
            }
            ratings_batch = self._run(fetches=self.model.outputs, feed_dict=feed_dict)
            ratings_batch = np.reshape(ratings_batch, (batch_user_num, batch_item_num))

            user_batch_rating_uid = zip(ratings_batch, uid_batch, user_click_item_train_batch, user_click_item_test_batch)
            batch_result = pool.map(test_one_user, user_batch_rating_uid)
            count += len(batch_result)

            for re in batch_result:
                result['recall'] += re['recall'] / n_test_users
                result['ndcg'] += re['ndcg'] / n_test_users

            print("count:", count, time.time() - tic)
            tic = time.time()
        print("end:", time.time() - first_time)
        pool.close()
        print(result)
        assert count == n_test_users
        return result

    def _batch_callback(self):
        pass

    def _epoch_callback(self,):
        tic = time.time()
        print('running test...')
        result = self.test_all_user()
        Ks = [5, 10, 20, 50, 100]  # matrix@k
        print('test output recall=[%.4f, %.4f, %.4f, %.4f, %.4f],ndcg=[%.4f, %.4f, %.4f, %.4f, %.4f]' % (
            result['recall'][0], result['recall'][1], result['recall'][2], result['recall'][3], result['recall'][4],
            result['ndcg'][0], result['ndcg'][1], result['ndcg'][2], result['ndcg'][3], result['ndcg'][4])
              )
        tmp_result = [result['recall'][1], result['ndcg'][1]]
        if self.max_result is None:
            self.max_result = tmp_result
            print("current max output:", tmp_result, "sum:", sum(tmp_result))
            self._save()
        elif sum(self.max_result) < sum(tmp_result):
            self.max_result = tmp_result
            print("current max output:", tmp_result, "sum:", sum(tmp_result))
            self._save()

        toc = time.time()
        print('evaluated time:', str(datetime.timedelta(seconds=int(toc - tic))))
        return

    def score(self):
        self._epoch_callback()

    def _save(self):
        if self.logdir is None:
            return
        var_list = []
        for variable in tf.global_variables():
            if "adam" not in variable.name.lower():
                var_list.append(variable)
        print('var_list:', var_list)
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=1)
        print('max saving checkpoint...', os.path.join(self.logdir, 'checkpoints', 'model.ckpt'),
              self.global_step.eval(self.session), 'save variable num:', len(var_list))
        if not os.path.exists(os.path.join(self.logdir, 'checkpoints')):
            os.makedirs(os.path.join(self.logdir, 'checkpoints'))
        self.saver.save(self.session, os.path.join(self.logdir, 'checkpoints', 'model.ckpt'),
                        self.global_step.eval(self.session))

    def fit(self):
        num_of_batches = int(np.ceil(self.train_per_epoch / self.batch_size))
        total_batches = self.n_epoch * num_of_batches
        print('total batches: %d\tbatch per epoch: %d' % (total_batches, num_of_batches))
        start_time = time.time()
        tic = time.time()
        epoch = 1
        finished_batches = 0
        avg_loss = 0
        avg_l2 = 0
        label_list = []
        pred_list = []
        test_every_epoch = self.test_every_epoch

        result = self.test_all_user()
        tmp_result = [result['recall'][1], result['ndcg'][1]]
        print("evaluate result")
        print(tmp_result)

        result2 = self.test_split_user()
        tmp_result2 = [result2['recall'][1], result2['ndcg'][1]]
        print("evaluate split result")
        print(tmp_result2)
        # exit(-1)
        return tmp_result
        