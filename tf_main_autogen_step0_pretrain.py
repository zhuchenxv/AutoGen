#encoding=utf-8
import sys
import time
import os
import __init__
sys.path.append(__init__.config['data_path'])
from Movielens import Movielens
from tf_trainers import Trainer
from tf_trainers_evaluate import Trainer_eval
from tf_utils import *
from tf_models import DSSM_multi_input
import tensorflow as tf
import traceback
import random
import numpy as np
seeds = [0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC,
             0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC]
backend = 'tf'


def run_one_model(dataset=None, model=None, learning_rate=1e-3, decay_rate=1.0, epsilon=1e-8, ep=5, logdir=None, pre_train_logdir=None,
                  max_epoch=None):
    n_ep = ep * max_epoch  # 40, 100
    train_param = {
        'opt': 'adam',
        'loss': 'weight',
        'pos_weight': 1.0,
        'n_epoch': n_ep,
        'train_per_epoch': dataset.train_size / ep,  # split training data
        'early_stop_epoch': int(0.5*ep),
        'batch_size': train_data_param['batch_size'],
        'learning_rate': learning_rate,
        'decay_rate': decay_rate,
        'epsilon': epsilon,
        'test_every_epoch': int(ep / 5),
        'logdir': logdir,
        'pre_train_logdir': pre_train_logdir
    }
    train_gen = dataset.batch_generator(train_data_param)
    # test_gen = dataset.batch_generator(test_data_param)
    test_gen = None
    trainer = Trainer(dataset=dataset, model=model, train_gen=train_gen, test_gen=test_gen, **train_param)
    trainer.fit()
    trainer.session.close()

def evaluate_one_model(dataset=None, model=None, learning_rate=1e-3, decay_rate=1.0, epsilon=1e-8, ep=5, logdir=None, pre_train_logdir=None,
                  max_epoch=None):
    n_ep = ep * max_epoch  # 40, 100
    train_param = {
        'opt': 'adam',
        'loss': 'weight',
        'pos_weight': 1.0,
        'n_epoch': n_ep,
        'train_per_epoch': dataset.train_size / ep,  # split training data
        'early_stop_epoch': int(0.5*ep),
        'batch_size': train_data_param['batch_size'],
        'learning_rate': learning_rate,
        'decay_rate': decay_rate,
        'epsilon': epsilon,
        'test_every_epoch': int(ep / 5),
        'logdir': logdir,
        'pre_train_logdir': pre_train_logdir
    }
    train_gen = dataset.batch_generator(train_data_param)
    # test_gen = dataset.batch_generator(test_data_param)
    test_gen = None
    trainer = Trainer_eval(dataset=dataset, model=model, train_gen=train_gen, test_gen=test_gen, **train_param)
    result = trainer.fit()
    trainer.session.close()
    return result


import math
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    print("run dssm_multi_input")
    random_seed = 2022
    batch_size = 2000
    user_multi_layer, item_multi_layer = True, True
    dc = 1.0
    max_epoch = 100  # 40, 100
    dropout_keep_rate = 0.6
    hidden_unit = [512, 256, 128]
    embedding_size = 40
    learning_rate = 5e-3
    initial_dropout = False
    l2_v = 0.0
    activation = "relu"
    user_dnn_hidden_unit = hidden_unit  # [512, 256, 128]
    item_dnn_hidden_unit = hidden_unit  # [512, 256, 128]
    depth = len(user_dnn_hidden_unit)
    user_dnn_activation = [activation] * (depth - 1) + [None]
    user_dnn_drop_rate = [dropout_keep_rate] * (depth)
    item_dnn_activation = [activation] * (depth - 1) + [None]
    item_dnn_drop_rate = [dropout_keep_rate] * (depth)
    batch_norm, layer_norm = True, False   # True, False; False, True; False, False
    initial_BN, initial_act = False, False
    similar_type = "inner"
    l2_layer = 0.0
    split_epoch = 1  # 5
    tf.reset_default_graph()
    all_random_seed(random_seed)
    train_data_param = {
        'part': 'train',
        'shuffle': True,
        'batch_size': batch_size,
    }
    logdir = None
    print("logdir:", logdir)
    dataset = Movielens()
    user_field_num = dataset.user_field_num
    item_field_num = dataset.item_field_num
    item_multi_field_num = dataset.item_multi_field_num
    user_feature_num = dataset.user_feature_num
    item_feature_num = dataset.item_feature_num
    model = DSSM_multi_input(init='xavier', user_field_num=user_field_num, item_field_num=item_field_num,
                 item_multi_field_num=item_multi_field_num,
                 user_feature_num=user_feature_num, item_feature_num=item_feature_num, embedding_size=embedding_size,
                 user_dnn_hidden_unit=user_dnn_hidden_unit, user_dnn_activation=user_dnn_activation,
                 user_dnn_drop_rate=user_dnn_drop_rate, use_bn=batch_norm, use_ln=layer_norm,
                 item_dnn_hidden_unit=item_dnn_hidden_unit, item_dnn_activation=item_dnn_activation,
                 item_dnn_drop_rate=item_dnn_drop_rate, similar_type=similar_type,
                 l2_v=l2_v, l2_layer=l2_layer,user_multi_layer=user_multi_layer, item_multi_layer=item_multi_layer,
                             initial_dropout=initial_dropout
                             )
    run_one_model(dataset=dataset, model=model, learning_rate=learning_rate, epsilon=1e-8,
                  decay_rate=dc, ep=split_epoch, logdir=logdir, max_epoch=max_epoch)
    tf.reset_default_graph()














