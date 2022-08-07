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

    def __init__(self, dataset=None, model=None, pre_train_logdir=None, save_logdir=None, select_structure_type=1,
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
        self.save_logdir = save_logdir

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

    def test_all_user(self, user_reserve_feature, user_reserve_embedding):
        Ks = [5, 10, 20, 50, 100]  # matrix@k
        result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}
        tic = time.time()
        all_test_item_one_hot, all_test_item_multi_hot = self.dataset.get_all_item()
        batch_size = 1000  # 1000
        test_data_param = {
            'part': 'validation',  # test, validation, new_test
            'shuffle': False,
            'batch_size': batch_size,
        }
        count = 0
        pool = multiprocessing.Pool(4)
        # n_test_users = self.dataset.all_user_num
        n_test_users = len(self.dataset.validation_user_order)
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
                self.model.user_reserve_feature: user_reserve_feature,
                self.model.user_reserve_embedding: user_reserve_embedding,
            }
            ratings_batch = self._run(fetches=self.model.final_ratings, feed_dict=feed_dict)

            user_batch_rating_uid = zip(ratings_batch, uid_batch, user_click_item_train_batch,
                                        user_click_item_test_batch)
            batch_result = pool.map(test_one_user, user_batch_rating_uid)
            count += len(batch_result)

            for re in batch_result:
                result['recall'] += re['recall'] / n_test_users
                result['ndcg'] += re['ndcg'] / n_test_users

            # print("count:", count, time.time() - tic)
            tic = time.time()
        # print("end:", time.time() - first_time)
        pool.close()
        # print(result)
        # exit(-1)
        assert count == n_test_users
        return result

    def _batch_callback(self):
        pass

    def fit(self):
        son_structure_sample_list = np.load('old/sample_son_structure.npy')  # 2000
        assert len(son_structure_sample_list) == 2000
        final_structure_result = dict()
        tmp_user_select_feature_matrix = np.ones(shape=(3, self.user_field_num))
        tmp_user_select_embedding_matrix = np.ones(shape=(self.user_field_num, 5))
        result = self.test_all_user(tmp_user_select_feature_matrix, tmp_user_select_embedding_matrix)
        print('test output recall_10=%.4f,ndcg_10=%.4f' % (result['recall'][1], result['ndcg'][1]))
        tmp_result = [result['recall'][1], result['ndcg'][1]]
        final_structure_result[(54, 90)] = tmp_result
        tic = time.time()
        self.argsort_alpha = np.argsort(-np.abs(self.alpha))  # 3, 18
        self.argsort_beta = np.argsort(-np.abs(self.beta))  # 18, 5
        son_structure_sample_list = son_structure_sample_list

        max_test_num = len(son_structure_sample_list)
        print("test structure num:", max_test_num, "max len:", 54 * 90)
        # print(son_structure_sample_list[:10])
        random_order = [i for i in range(max_test_num)]
        random.shuffle(random_order)
        son_structure_sample_list = son_structure_sample_list[random_order]
        # print(son_structure_sample_list[:10])
        for test_i in range(max_test_num):
            tmp_user_select_feature_num = son_structure_sample_list[test_i][0]
            tmp_user_select_embedding_num = son_structure_sample_list[test_i][1]
            if (tmp_user_select_feature_num, tmp_user_select_embedding_num) in final_structure_result:
                print("exist before")
                continue
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

            # print(tmp_user_select_feature_num, tmp_user_select_embedding_num)
            # print(tmp_user_select_feature_matrix)
            # print(tmp_user_select_embedding_matrix)
            # exit(-1)

            result = self.test_all_user(tmp_user_select_feature_matrix, tmp_user_select_embedding_matrix)
            print("test_i:", test_i, ",tmp_user_select_num:", tmp_user_select_feature_num, tmp_user_select_embedding_num)
            print('test output recall_10=%.4f,ndcg_10=%.4f' % (result['recall'][1], result['ndcg'][1])
                  , tmp_user_select_feature_num, tmp_user_select_embedding_num)
            tmp_result = [result['recall'][1], result['ndcg'][1]]
            if self.max_result is None or sum(tmp_result) > sum(self.max_result):
                self.max_result = tmp_result
                self.max_result_structure = (tmp_user_select_feature_num, tmp_user_select_embedding_num)
                print("max result:", self.max_result, "max structure:", self.max_result_structure)
            final_structure_result[(tmp_user_select_feature_num, tmp_user_select_embedding_num)] = tmp_result
            if self.save_logdir is not None:
                if test_i % 10 == 0:
                    np.save(self.save_logdir, final_structure_result)
            print("spent time:%s" % str(datetime.timedelta(seconds=int(time.time() - tic))))
        np.save(self.save_logdir, final_structure_result)
        print(len(final_structure_result))
        # print(final_structure_result)
        final_structure_result_sum = {}
        for son_structure in final_structure_result:
            final_structure_result_sum[son_structure] = sum(final_structure_result[son_structure])
        topK_final_structure_result_sum = sorted(final_structure_result_sum.items(), key=lambda x: x[1], reverse=True)
        print("top10 structure")
        for k in range(10):
            print("top", k, "structure:", topK_final_structure_result_sum[k][0],
                  "result:", final_structure_result[topK_final_structure_result_sum[k][0]],
                  "sum:", topK_final_structure_result_sum[k][1])
        return
