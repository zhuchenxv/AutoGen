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
                 test_every_epoch=1, logdir=None):
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

    def _run(self, fetches, feed_dict):
        return self.session.run(fetches=fetches, feed_dict=feed_dict)

    def _train(self, user_x, item_one_hot_x, item_multi_hot_x, y):
        feed_dict = {
            self.model.labels: y,
            self.learning_rate: self._learning_rate,
            self.model.user_input: user_x,
            self.model.item_input: item_one_hot_x,
            self.model.item_multi_hot_input: item_multi_hot_x[:, :-1],
            self.model.item_multi_hot_input_len: item_multi_hot_x[:, -1:],
            self.model.training: True
        }
        if self.model.l2_loss is None:
            _, _loss, outputs = self._run(fetches=[self.model.optimizer, self.model.loss, self.model.outputs],
                                              feed_dict=feed_dict)
            _l2_loss = 0
        else:
            _, _loss, _l2_loss, outputs = self._run(
                    fetches=[self.model.optimizer, self.model.loss, self.model.l2_loss,
                             self.model.outputs], feed_dict=feed_dict)
        return _loss, _l2_loss, outputs

    def _predict(self, user_x, item_one_hot_x, item_multi_hot_x, y):
        feed_dict = {
            self.model.labels: y,
            self.model.user_input: user_x,
            self.model.item_input: item_one_hot_x,
            self.model.item_multi_hot_input: item_multi_hot_x[:, :-1],
            self.model.item_multi_hot_input_len: item_multi_hot_x[:, -1:],
            self.model.training: False
        }
        return self._run(fetches=[self.model.loss, self.model.outputs], feed_dict=feed_dict)

    def predict(self, gen, eval_size):
        hits = []
        ndcgs = []
        cnt = 0
        tic = time.time()
        for batch_data in gen:
            user_x, item_one_hot_x, item_multi_hot_x, y = batch_data
            _, batch_pred = self._predict(user_x, item_one_hot_x, item_multi_hot_x, y)
            assert user_x.shape[0] % 101 == 0
            for user_i in range(int(user_x.shape[0]/101)):
                items = item_one_hot_x[user_i*101:(user_i+1)*101, 0]
                tmp_y = y[user_i*101:(user_i+1)*101]
                tmp_batch_pred = batch_pred[user_i*101:(user_i+1)*101]
                right_item = items[-1]
                assert tmp_y[-1] == 1
                map_item_score = {}
                for i in range(len(items)):
                    item = items[i]
                    map_item_score[item] = tmp_batch_pred[i]
                ranklist = heapq.nlargest(10, map_item_score, key=map_item_score.get)
                hr = getHitRatio(ranklist, right_item)
                ndcg = getNDCG(ranklist, right_item)

                hits.append(hr)
                ndcgs.append(ndcg)
                cnt += 1
            if cnt % 3000 == 0:
                print('evaluated batches:', cnt, time.time() - tic)
                tic = time.time()
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        return hr, ndcg

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
        print("analyse_structure")
        self.model.analyse_structure(self.session, print_full_weight=True)
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
        self.model.analyse_structure(self.session, print_full_weight=False)
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

        self._epoch_callback()
        while epoch <= self.n_epoch:
            print('new iteration')
            epoch_batches = 0

            for batch_data in self.train_gen:
                user_x, item_one_hot_x, item_multi_hot_x, y = batch_data
                label_list.append(y)
                batch_loss, batch_l2, batch_pred = self._train(user_x, item_one_hot_x, item_multi_hot_x, y)
                pred_list.append(batch_pred)
                avg_loss += batch_loss
                avg_l2 += batch_l2
                finished_batches += 1
                epoch_batches += 1

                if epoch_batches % 100 == 0:
                    avg_loss /= 100
                    avg_l2 /= 100
                    label_list = np.concatenate(label_list)
                    pred_list = np.concatenate(pred_list)
                    moving_auc = self.call_auc(y_true=label_list, y_score=pred_list)
                    elapsed = int(time.time() - start_time)
                    eta = int((total_batches - finished_batches) / finished_batches * elapsed)
                    print("elapsed : %s, ETA : %s" % (str(datetime.timedelta(seconds=elapsed)),
                                                      str(datetime.timedelta(seconds=eta))))
                    print('epoch %d / %d, batch %d / %d, global_step = %d, learning_rate = %e, loss = %f, l2 = %f, '
                          'auc = %f' % (epoch, self.n_epoch, epoch_batches, num_of_batches,
                                        self.global_step.eval(self.session), self._learning_rate,
                                        avg_loss, avg_l2, moving_auc))
                    label_list = []
                    pred_list = []
                    avg_loss = 0
                    avg_l2 = 0

                if epoch_batches % num_of_batches == 0:
                    # if epoch % test_every_epoch == 0:
                    self._epoch_callback()
                    # exit(-1)
                    self._learning_rate *= self.decay_rate
                    epoch += 1
                    epoch_batches = 0
                    if epoch > self.n_epoch:
                        return

            if epoch_batches % num_of_batches != 0 and epoch_batches % num_of_batches > 0.5 * num_of_batches:
                # if epoch % test_every_epoch == 0:
                self._epoch_callback()
                self._learning_rate *= self.decay_rate
                epoch += 1
                epoch_batches = 0
                if epoch > self.n_epoch:
                    return



