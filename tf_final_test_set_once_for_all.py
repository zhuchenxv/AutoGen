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

    def __init__(self, dataset=None, model=None, pre_train_logdir=None, son_structure_performance_logdir=None, select_structure_type=1,
                 alpha=None, beta=None, save_key=None, evaluate_way=None):
        self.model = model
        self.dataset = dataset
        self.max_result = None
        self.max_result_structure = None
        self.user_field_num = self.dataset.user_field_num
        self.item_field_num = self.dataset.item_field_num + self.dataset.item_multi_field_num
        self.alpha = alpha
        self.beta = beta
        self.evaluate_way = evaluate_way
        print("current max output:", 0)

        self.call_auc = roc_auc_score
        self.call_loss = log_loss
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        tf.summary.scalar('global_step', self.global_step)

        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())
        self.select_structure_type = select_structure_type
        self.son_structure_performance_logdir = son_structure_performance_logdir

        if pre_train_logdir is not None:
            print(os.path.join(pre_train_logdir, save_key))
            module_file = tf.train.latest_checkpoint(os.path.join(pre_train_logdir, save_key))
            # tf.train.Saver(max_to_keep=1).restore(self.session, module_file)
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

    def test_split_user(self, user_reserve_feature_list, user_reserve_embedding_list):
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
            # print("part user:", split_user, "split_result:", split_result)
            # print(np.sum(user_reserve_feature_list[split_user]), np.sum(user_reserve_embedding_list[split_user]))
            for batch_data in self.dataset.batch_generator(test_data_param):
                # user_feature_batch, user_click_item_train_batch, user_click_item_test_batch, user_value_batch = batch_data
                user_feature_batch, user_click_item_train_batch, user_click_item_test_batch = batch_data
                uid_batch = user_feature_batch[:, 0]
                # print(user_feature_batch.shape)
                current_batch_size = uid_batch.shape[0]
                feed_dict = {
                    self.model.user_input: user_feature_batch,
                    self.model.item_input: all_test_item_one_hot,
                    self.model.item_multi_hot_input: all_test_item_multi_hot[:, :-1],
                    self.model.item_multi_hot_input_len: all_test_item_multi_hot[:, -1:],
                    self.model.training: False,
                    self.model.user_reserve_feature: user_reserve_feature_list[split_user],
                    self.model.user_reserve_embedding: user_reserve_embedding_list[split_user],
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

    def test_all_user(self, user_reserve_feature, user_reserve_embedding):
        Ks = [5, 10, 20, 50, 100]  # matrix@k
        result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}
        tic = time.time()
        all_test_item_one_hot, all_test_item_multi_hot = self.dataset.get_all_item()
        batch_size = 1000  # 1000
        n_test_users = len(self.dataset.only_test_user_order)
        # n_test_users = self.dataset.all_user_num
        count = 0
        pool = multiprocessing.Pool(4)

        test_data_param = {
            'part': 'new_test',  # test_old, new_test
            'shuffle': False,
            'batch_size': batch_size,
        }
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
                self.model.user_reserve_feature: user_reserve_feature,
                self.model.user_reserve_embedding: user_reserve_embedding,
            }
            ratings_batch = self._run(fetches=self.model.final_ratings, feed_dict=feed_dict)

            user_batch_rating_uid = zip(ratings_batch, uid_batch, user_click_item_train_batch,
                                        user_click_item_test_batch)
            batch_result = pool.map(test_one_user, user_batch_rating_uid)

            for re in batch_result:
                result['recall'] += re['recall'] / n_test_users
                result['ndcg'] += re['ndcg'] / n_test_users
            count += current_batch_size
        pool.close()
        assert count == n_test_users
        return result

    def _batch_callback(self):
        pass

    def fit(self):
        # 检验 [0-28] & [0-28] & [0-28]
        final_result = []
        son_structure_sample_list = np.load('old/sample_son_structure.npy')  # 2000
        son_structure_performance_dict = np.load(self.son_structure_performance_logdir, allow_pickle=True).item()
        son_structure_flop_dict = np.load('old/son_structure_flop.npy', allow_pickle=True).item()
        self.argsort_alpha = np.argsort(-np.abs(self.alpha))  # 3, 18
        self.argsort_beta = np.argsort(-np.abs(self.beta))  # 18, 5
        dssm_flop_num = 705164
        max_network_flop_num = 1640084

        print("get once_for_all top10 network")

        min_flop_num = dssm_flop_num * 0.6
        max_flop_num = dssm_flop_num * 1.5

        strict_flop_list = []
        for strict_i in range(9, -1, -1):
            tmp_flop_num = min_flop_num + int((max_flop_num - min_flop_num) * strict_i / 9)
            strict_flop_list.append(tmp_flop_num)
        print(max_flop_num)
        print(min_flop_num)
        print(strict_flop_list)
        print(len(strict_flop_list))
        avg_flop_num = 0

        all_son_structure_performance_dict = []
        for key in son_structure_performance_dict:
            tmp_tuple = (key, son_structure_performance_dict[key][0], son_structure_flop_dict[key])
            all_son_structure_performance_dict.append(tmp_tuple)
        all_son_structure_performance_sorted_list = sorted(all_son_structure_performance_dict,
                                                            key=lambda x: (-x[1], x[2], x[0][0], x[0][1]))
        user_reserve_feature_list, user_reserve_embedding_list = [], []
        son_structure_flop_list = []
        for strict_i in range(10):
            tmp_structure_top1 = None
            for structure_i in range(len(all_son_structure_performance_sorted_list)):
                tmp_structure, tmp_performance, tmp_flop = all_son_structure_performance_sorted_list[structure_i]
                if son_structure_flop_dict[tmp_structure] <= strict_flop_list[strict_i]:
                    tmp_tuple = (
                    tmp_structure[0], sum(tmp_structure[1:]), son_structure_performance_dict[tmp_structure],
                    son_structure_flop_dict[tmp_structure])
                    tmp_structure_top1 = tmp_tuple
                    break
            if tmp_structure_top1 is None:
                print("wrong")
                exit(-1)
            print("son structure:", tmp_structure_top1)
            son_structure_flop_list.append(tmp_structure_top1[-1])
            avg_flop_num += tmp_structure_top1[-1]
            tmp_user_select_feature_num = tmp_structure_top1[0]
            tmp_user_select_embedding_num = tmp_structure_top1[1]
            tmp_user_select_feature_matrix = np.zeros(shape=(54))
            for j in range(tmp_user_select_feature_num):
                tmp_user_select_feature_matrix[self.argsort_alpha[j]] = 1
            tmp_user_select_feature_matrix = np.reshape(tmp_user_select_feature_matrix, (3, 18))

            tmp_user_select_embedding_matrix = np.zeros(shape=(18, 5))
            user_embedding_len = np.zeros(shape=(18), dtype=np.int32)
            remain_num = tmp_user_select_embedding_num
            for i in range(remain_num):
                tmp_feature = self.argsort_beta[i]
                user_embedding_len[int(tmp_feature / 5)] += 1
            for i in range(18):
                tmp_user_select_embedding_matrix[i, 0:user_embedding_len[i]] = 1
            user_reserve_feature_list.append(tmp_user_select_feature_matrix)
            user_reserve_embedding_list.append(tmp_user_select_embedding_matrix)
        for strict_i in range(10):
            print(son_structure_flop_list[strict_i])
        result = self.test_split_user(user_reserve_feature_list, user_reserve_embedding_list)
        print('test output recall=[%.4f, %.4f, %.4f, %.4f, %.4f],ndcg=[%.4f, %.4f, %.4f, %.4f, %.4f]' % (
            result['recall'][0], result['recall'][1], result['recall'][2], result['recall'][3], result['recall'][4],
            result['ndcg'][0], result['ndcg'][1], result['ndcg'][2], result['ndcg'][3], result['ndcg'][4])
                )
        final_result.append([result['recall'][0], result['recall'][1], result['recall'][2], result['recall'][3],
                                result['recall'][4],
                                result['ndcg'][0], result['ndcg'][1], result['ndcg'][2], result['ndcg'][3],
                                result['ndcg'][4],
                                (avg_flop_num / 10)])
        final_result = np.round(final_result, 4)
        final_result = final_result[:, [1, 6, 10]]
        print("final_result:")
        for i in range(len(final_result)):
            print(*final_result[i], sep=',')
        
        return final_result