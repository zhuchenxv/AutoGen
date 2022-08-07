from __future__ import division
from __future__ import print_function

import datetime
import os
import time
import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, log_loss
from tf_utils import get_optimizer, get_loss
from tensorflow.python import pywrap_tensorflow
import heapq
import math
from test_matrix import test_one_user
import random
import multiprocessing

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


class Trainer:
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
                 test_every_epoch=1, pre_train_logdir=None, save_logdir=None,
                 select_structure_type=1, change_structure_batch=100, select_son_ratio_train=1,
                 alpha=None, beta=None, optimizer_type=None, epoch_add_select_step=1, step_add_num=1):
        self.model = model
        self.dataset = dataset
        self.train_gen = train_gen
        self.test_gen = test_gen
        self.valid_gen = valid_gen
        optimizer = get_optimizer(opt)
        loss = get_loss(loss)
        self.pos_weight = pos_weight
        self.n_epoch = n_epoch
        self.train_per_epoch = train_per_epoch
        self.early_stop_epoch = early_stop_epoch
        self.test_per_epoch = test_per_epoch
        self.batch_size = batch_size
        self._learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.test_every_epoch = test_every_epoch
        self.max_result = None
        self.max_result_after_20 = None
        self.max_result_after_40 = None
        self.max_result_after_search = None
        self.user_field_num = self.dataset.user_field_num
        self.item_field_num = self.dataset.item_field_num + self.dataset.item_multi_field_num
        self.select_son_ratio_train = select_son_ratio_train
        self.alpha = np.abs(alpha)
        self.beta = np.abs(beta)
        self.beta_num = 8
        self.alpha_size = self.alpha.shape[0]
        self.beta_size = self.beta.shape[0]
        # print(self.alpha.shape)
        # print(self.beta.shape)
        # print(self.user_field_num)
        assert self.alpha.shape[0] == (self.user_field_num * 3)
        assert (self.beta.shape[0] % self.user_field_num) == 0
        # assert self.beta.shape[1] == self.user_field_num
        self.user_search_embed_block_num = int(self.beta.shape[0] / self.user_field_num)
        self.optimizer_type = optimizer_type
        self.epoch_add_select_step = epoch_add_select_step
        self.step_add_num = step_add_num
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
            opt = optimizer(learning_rate=self.learning_rate, epsilon=self.epsilon)  # TODO fbh
        elif opt == 'adagrad':
            opt = optimizer(learning_rate=self.learning_rate, initial_accumulator_value=initial_accumulator_value)
        elif opt == 'moment':
            opt = optimizer(learning_rate=self.learning_rate, momentum=momentum)
        else:
            opt = optimizer(learning_rate=self.learning_rate, )  # TODO fbh

        self.model.compile(loss=loss, optimizer=opt, global_step=self.global_step, pos_weight=pos_weight)

        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())
        self.select_structure_type = select_structure_type
        self.change_structure_batch = change_structure_batch

        self.save_logdir = save_logdir
        if pre_train_logdir is not None:
            module_file = tf.train.latest_checkpoint(pre_train_logdir)
            print(module_file)
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

    def _train(self, user_x, item_one_hot_x, item_multi_hot_x, y, user_select_num_list, user_embedding_num_select_list,
               train_sub=True):
        for i in range(len(user_select_num_list)):
            feed_dict = {
                self.model.labels: y,
                self.learning_rate: self._learning_rate / len(user_select_num_list),
                self.model.user_input: user_x,
                self.model.item_input: item_one_hot_x,
                self.model.item_multi_hot_input: item_multi_hot_x[:, :-1],
                self.model.item_multi_hot_input_len: item_multi_hot_x[:, -1:],
                self.model.training: True,
                self.model.user_reserve_feature: user_select_num_list[i],
                self.model.user_reserve_embedding: user_embedding_num_select_list[i],
            }
            _, _loss, outputs, _l2_loss = self._run(
                fetches=[self.model.optimizer, self.model.loss, self.model.outputs, self.model.l2_loss],
                feed_dict=feed_dict)
            return _loss, _l2_loss, outputs


    def test_all_user(self):
        Ks = [5, 10, 20, 50, 100]  # matrix@k
        result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}
        tic = time.time()
        all_test_item_one_hot, all_test_item_multi_hot = self.dataset.get_all_item()
        batch_size = 1000  # 1000
        test_data_param = {
            'part': 'new_test',  # test, validation, new_test
            'shuffle': False,
            'batch_size': batch_size,
        }
        count = 0
        pool = multiprocessing.Pool(4)
        n_test_users = len(self.dataset.only_test_user_order)
        print("begin test_users:", n_test_users)
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
                self.model.user_reserve_feature: np.ones(shape=[3, self.user_field_num]),
                self.model.user_reserve_embedding: np.ones(shape=[self.user_field_num,
                                                                  self.user_search_embed_block_num]),
            }
            ratings_batch = self._run(fetches=self.model.final_ratings, feed_dict=feed_dict)

            user_batch_rating_uid = zip(ratings_batch, uid_batch, user_click_item_train_batch,
                                        user_click_item_test_batch)
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

    def _batch_callback(self):
        pass

    def _epoch_callback(self, epoch=None):
        tic = time.time()
        print('running test...')
        result = self.test_all_user()
        Ks = [5, 10, 20, 50, 100]  # matrix@k
        print('test output recall=[%.4f, %.4f, %.4f, %.4f, %.4f],ndcg=[%.4f, %.4f, %.4f, %.4f, %.4f]' % (
            result['recall'][0], result['recall'][1], result['recall'][2], result['recall'][3], result['recall'][4],
            result['ndcg'][0], result['ndcg'][1], result['ndcg'][2], result['ndcg'][3], result['ndcg'][4])
              )

        # exit(-1)
        tmp_result = [result['recall'][1], result['ndcg'][1]]
        if self.max_result is None:
            self.max_result = tmp_result
            print("current max output:", tmp_result, "sum:", sum(tmp_result), "current epoch:", epoch)
            self._save(key='max_checkpoints')
        elif sum(self.max_result) < sum(tmp_result):
            self.max_result = tmp_result
            print("current max output:", tmp_result, "sum:", sum(tmp_result), "current epoch:", epoch)
            self._save(key='max_checkpoints')
        self._save(key='last_checkpoints')
        if self.epoch >= int(40/self.epoch_add_select_step/self.step_add_num) or self.select_structure_type == 5:
            if self.max_result_after_search is None:
                self.max_result_after_search = tmp_result
                print("after search current max output:", tmp_result, "sum:", sum(tmp_result), "current epoch:", epoch)
                self._save(key='max_checkpoints_after_search')
            elif sum(self.max_result_after_search) < sum(tmp_result):
                self.max_result_after_search = tmp_result
                print("after search current max output:", tmp_result, "sum:", sum(tmp_result), "current epoch:", epoch)
                self._save(key='max_checkpoints_after_search')
        toc = time.time()
        print('evaluated time:', str(datetime.timedelta(seconds=int(toc - tic))))
        return

    def score(self):
        self._epoch_callback()

    def _save(self, key='checkpoints'):
        if self.save_logdir is None:
            return
        var_list = []
        for variable in tf.global_variables():
            if "adam" not in variable.name.lower():
                var_list.append(variable)
        print('var_list:', len(var_list))
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=1)
        print('saving ', key, os.path.join(self.save_logdir, key, 'model.ckpt'),
              self.global_step.eval(self.session), 'save variable num:', len(var_list))
        if not os.path.exists(os.path.join(self.save_logdir, key)):
            os.makedirs(os.path.join(self.save_logdir, key))
        self.saver.save(self.session, os.path.join(self.save_logdir, key, 'model.ckpt'),
                                           self.global_step.eval(self.session))

    def fit(self):
        num_of_batches = int(np.ceil(self.train_per_epoch / self.batch_size))
        total_batches = self.n_epoch * num_of_batches
        print('total batches: %d\tbatch per epoch: %d' % (total_batches, num_of_batches))
        add_step_batches = int(self.train_per_epoch / self.batch_size / self.epoch_add_select_step)
        start_time = time.time()
        tic = time.time()
        self.epoch = 1
        finished_batches = 0
        avg_loss = 0
        avg_l2 = 0
        label_list = []
        pred_list = []
        test_every_epoch = self.test_every_epoch
        self._epoch_callback()
        # exit(-1)
        current_batch = 0
        self.select_step = 0  # 0->1->2->...->19->...->39
        self.remain_feature_num_different_step_list = []  # 40 阶段
        self.remain_embedding_different_step_list = []
        # 合起来排序
        self.argsort_alpha = np.argsort(-self.alpha)  # 3, 18
        self.argsort_beta = np.argsort(-self.beta)  # 18, 10
        if self.select_structure_type == 1:
            # first global field, second global embedding
            for j in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1,
                      0.05, 0.0]:
                tmp_remain_feature = [num for num in range(int(j * self.alpha_size), self.alpha_size + 1)]
                tmp_remain_embedding = [self.beta_size]
                self.remain_feature_num_different_step_list.append(tmp_remain_feature)
                self.remain_embedding_different_step_list.append(tmp_remain_embedding)
            for j in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1,
                      0.05, 0.0]:
                tmp_remain_feature = [num for num in range(0, self.alpha_size + 1)]
                tmp_remain_embedding = [num for num in range(int(j * self.beta_size), self.beta_size + 1)]
                self.remain_feature_num_different_step_list.append(tmp_remain_feature)
                self.remain_embedding_different_step_list.append(tmp_remain_embedding)
        elif self.select_structure_type == 5:
            for j in range(40):
                self.remain_feature_num_different_step_list.append([self.alpha_size])
                self.remain_embedding_different_step_list.append([self.beta_size])
        else:
            print("please print right select_structure_type")
            exit(-1)
        while self.epoch <= self.n_epoch:
            print('new iteration')
            print("-----------------------------------------------------------------------------------------------")
            print("current epoch:", self.epoch, "current_step:", self.select_step,
                  "select_num_list:",
                  self.remain_feature_num_different_step_list[int(self.select_step)],
                  self.remain_embedding_different_step_list[int(self.select_step)])
            epoch_batches = 0
            for batch_data in self.train_gen:
                user_x, item_one_hot_x, item_multi_hot_x, y = batch_data
                label_list.append(y)
                if current_batch % self.change_structure_batch == 0:
                    # item_select_num = np.ones(shape=(3, self.item_field_num))
                    num_son_structure = 1
                    user_select_feature_list, user_select_embedding_list = [], []
                    for son_i in range(num_son_structure):
                        tmp_select_feature_num_list = self.remain_feature_num_different_step_list[int(self.select_step)]
                        tmp_select_embedding_num_list = self.remain_embedding_different_step_list[int(self.select_step)]
                        # print(tmp_select_feature_num_list)
                        # print(tmp_select_embedding_num_list)

                        tmp_user_select_feature_matrix = np.zeros(shape=(self.alpha_size))
                        tmp_user_select_feat_num = random.choice(tmp_select_feature_num_list)
                        for j in range(tmp_user_select_feat_num):
                            tmp_user_select_feature_matrix[self.argsort_alpha[j]] = 1
                        tmp_user_select_feature_matrix = np.reshape(tmp_user_select_feature_matrix,
                                                                    (3, self.user_field_num))

                        tmp_user_select_embedding_matrix = np.zeros(shape=(self.user_field_num, self.user_search_embed_block_num))
                        tmp_user_embedding_len = np.zeros(shape=(self.user_field_num), dtype=np.int32)
                        tmp_user_select_embedding_num = random.choice(tmp_select_embedding_num_list)
                        for j in range(tmp_user_select_embedding_num):
                            tmp_user_embedding_len[int(self.argsort_beta[j] / self.user_search_embed_block_num)] += 1
                        for j in range(self.user_field_num):
                            tmp_user_select_embedding_matrix[j, 0:tmp_user_embedding_len[j]] = 1

                        if current_batch % 100 == 0 and son_i == 0:
                            # print(tmp_user_select_feature_matrix)
                            # print(tmp_user_select_embedding_matrix)
                            print("sample structure1, feature num:", np.sum(tmp_user_select_feature_matrix),
                                  "embedding num:", np.sum(tmp_user_select_embedding_matrix),
                                  np.min(tmp_user_embedding_len),
                                  np.max(tmp_user_embedding_len))
                        user_select_feature_list.append(tmp_user_select_feature_matrix)
                        user_select_embedding_list.append(tmp_user_select_embedding_matrix)


                batch_loss, batch_l2, batch_pred = \
                    self._train(user_x, item_one_hot_x, item_multi_hot_x, y, user_select_feature_list,
                                user_select_embedding_list, train_sub=True)

                pred_list.append(batch_pred)
                avg_loss += batch_loss
                avg_l2 += batch_l2
                finished_batches += 1
                epoch_batches += 1
                current_batch += 1

                epoch_batches_num = 100
                if epoch_batches % epoch_batches_num == 0:
                    avg_loss /= epoch_batches_num
                    avg_l2 /= epoch_batches_num
                    label_list = np.concatenate(label_list)
                    pred_list = np.concatenate(pred_list)
                    moving_auc = self.call_auc(y_true=label_list, y_score=pred_list)
                    elapsed = int(time.time() - start_time)
                    eta = int((total_batches - finished_batches) / finished_batches * elapsed)
                    print("elapsed : %s, ETA : %s" % (str(datetime.timedelta(seconds=elapsed)),
                                                      str(datetime.timedelta(seconds=eta))))
                    print('epoch %d / %d, batch %d / %d, global_step = %d, learning_rate = %e, loss = %f, auc = %f' %
                          (self.epoch, self.n_epoch, epoch_batches, num_of_batches,
                                        self.global_step.eval(self.session), self._learning_rate,
                                        avg_loss, moving_auc))
                    label_list = []
                    pred_list = []
                    avg_loss = 0
                    avg_l2 = 0

                if epoch_batches % add_step_batches == 0:
                    if self.select_step < 39:
                        self.select_step += self.step_add_num
                    if self.select_step >= 39:
                        self.select_step = 39
                    print("111 current_step:", self.select_step, "select_num_list:",
                          self.remain_feature_num_different_step_list[int(self.select_step)],
                          self.remain_embedding_different_step_list[int(self.select_step)])

            # if epoch_batches % num_of_batches == 0:
            self._epoch_callback(self.epoch)
            self._learning_rate *= self.decay_rate
            self.epoch += 1
            if epoch_batches % add_step_batches != 0 and epoch_batches % add_step_batches > 0.5 * add_step_batches:
                if self.select_step < 39:
                    self.select_step += self.step_add_num
                if self.select_step >= 39:
                    self.select_step = 39
                print("222 current_step:", self.select_step, "select_num_list:",
                      self.remain_feature_num_different_step_list[int(self.select_step)],
                      self.remain_embedding_different_step_list[int(self.select_step)])

            epoch_batches = 0
            if self.epoch > self.n_epoch:
                return



