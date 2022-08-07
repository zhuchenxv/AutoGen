#encoding=utf-8
import sys
import time
import os
import __init__
# from handler_dataset import Movielens
import numpy as np
sys.path.append(__init__.config['data_path'])
from Movielens import Movielens
from tf_final_test_set_once_for_all import Trainer
from tf_models import DSSM_multi_input_once_for_all
import tensorflow as tf
from tf_utils import *
import traceback
import socket
seeds = [0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC,
             0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC]
backend = 'tf'


def run_one_model(dataset=None, model=None,pre_train_logdir=None, son_structure_performance_logdir=None,
                  select_structure_type=1, alpha=None, beta=None, save_key=None, evaluate_way=None):
    train_param = {
        'pre_train_logdir': pre_train_logdir,
        'son_structure_performance_logdir': son_structure_performance_logdir,
        'select_structure_type': select_structure_type,
        'alpha': alpha,
        'beta': beta,
        'save_key': save_key,
        'evaluate_way': evaluate_way
    }
    trainer = Trainer(dataset=dataset, model=model, **train_param)
    result = trainer.fit()
    trainer.session.close()
    return result


import math
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

alpha = [0.39714313,0.16780508,0.0005186269,-0.00917678,0.3416384,-0.02242239,0.029514661,0.44124347,0.072089575,0.2662178,0.4426606,0.02585494,0.11951867,0.257596,0.3112677,0.007339239,0.3598279,0.030321935,0.27955294,0.014635027,-0.0006132904,-0.029932514,0.012801649,-0.014725087,-0.02262849,-0.03930172,-0.04083944,0.046517976,0.014259753,-0.012162334,0.05167715,0.0040545147,-0.003696015,0.011884609,0.057643216,-0.008003109,0.022150971,0.0068186997,0.009444783,0.00548271,0.009088712,0.0032386137,0.007903488,0.0028203395,0.0015112135,0.0034217522,0.007061683,-0.00074605,-0.0022163545,0.0054165856,-0.0027527162,-0.0031293896,0.012429041,0.0019685046]
beta = [0.034127895,0.033376083,0.035600435,0.030249925,0.036311544,-0.014257971,0.047756534,-0.011828341,-0.034527633,0.015101746,0.0005371022,0.06645153,0.015395728,0.020652054,-0.025991265,-0.004277289,0.0039948607,0.008576264,-0.0121924225,-0.020941442,0.014709744,0.012384741,0.01867885,0.016442234,0.018323518,0.007971605,0.008592229,-0.016125016,0.0071878238,-0.0018872068,-0.0034470966,-0.0067661763,-0.0001335303,0.015579359,0.014439113,0.016446797,0.009874817,0.017658722,0.020354703,0.014072323,0.016705181,-0.0025958254,0.01353705,0.0059331045,-0.0018090425,0.012940373,0.011589683,0.014106541,0.00732168,0.012095058,0.011690783,0.017303353,0.016023288,0.019694626,0.014168393,0.022271926,-0.0019669798,-0.0055782897,-0.0017618096,-0.005648979,0.0022232202,-0.0027179036,0.003921953,0.00897048,0.014818262,0.01579618,0.006411127,0.00818153,0.010983045,0.0026539934,-0.008017432,0.013552749,0.015796902,-0.0035538473,0.003152931,0.0106271645,0.010713519,0.0060347603,-0.0039533083,0.010702854,0.02181484,0.026933804,0.022118941,0.019825052,0.020798823,-0.0018506732,0.0036481014,-0.0012732466,-0.0026528533,0.010365783]


if __name__ == "__main__":
    all_result = []
    random_seed = 2022
    alpha = np.abs(alpha)
    beta = np.abs(beta)
    pre_train = True
    optimizer_type = 4
    l2_v = 0.0
    l2_layer = 0.0
    epoch_add_select_step = 1  # 1个epoch step更换几次

    step_add_num = 1.0  # 1, 0.5
    embedding_size, pre_train_i = 40, 0  # 40, 80

    select_structure_type = 1
    initial_dropout, special_dropout = True, False
    change_structure_batch = 10
    drop_keep_rate = 0.8

    learning_rate = 5e-4
    batch_size = 2000
    dc = 1.0
    share_embedding = True  # True, False
    user_multi_input, item_multi_input = True, True
    batch_norm = True
    layer_norm = False
    user_dnn_hidden_unit = [512, 256, 128]
    item_dnn_hidden_unit = [512, 256, 128]
    user_dnn_activation = ["relu", "relu", None]
    user_dnn_drop_rate = [drop_keep_rate, drop_keep_rate, drop_keep_rate]
    item_dnn_activation = ["relu", "relu", None]
    item_dnn_drop_rate = [drop_keep_rate, drop_keep_rate, drop_keep_rate]
    similar_type = "inner"
    split_epoch = 1  # 5
    all_random_seed(random_seed)
    train_data_param = {
        'part': 'train',
        'shuffle': True,
        'batch_size': batch_size,
    }
    gpu_name = socket.getfqdn(socket.gethostname())
    save_key = 'max_checkpoints_after_search'
    pre_train_logdir = '../../save_model/Movielens/once_for_all/search/' \
                'model_step_add_%f_k%d_pre%d_type%d_change_batch%d_drop%d%d_keep_rate_%f_lr_%f_random_%d' % \
                (step_add_num, embedding_size, pre_train_i, select_structure_type, change_structure_batch,
                initial_dropout, special_dropout, drop_keep_rate, learning_rate, random_seed)

    importance_way = 3
    son_structure_performance_logdir = '../../save_model/Movielens/once_for_all/evaluate_son/' \
                                        'son_type_%d_ini%d%d_change_batch_%d_drop_%f_random_%d.npy' % \
                                        (select_structure_type, initial_dropout, special_dropout,
                                        change_structure_batch, drop_keep_rate, random_seed)
    evaluate_way = None
    dataset = Movielens(importance_way=importance_way)
    user_field_num = dataset.user_field_num
    item_field_num = dataset.item_field_num
    item_multi_field_num = dataset.item_multi_field_num
    user_feature_num = dataset.user_feature_num
    item_feature_num = dataset.item_feature_num
    model = DSSM_multi_input_once_for_all(init='xavier', user_field_num=user_field_num,
                                        item_field_num=item_field_num,
                                        item_multi_field_num=item_multi_field_num,
                                        user_feature_num=user_feature_num, item_feature_num=item_feature_num,
                                        embedding_size=embedding_size,
                                        user_dnn_hidden_unit=user_dnn_hidden_unit,
                                        user_dnn_activation=user_dnn_activation,
                                        user_dnn_drop_rate=user_dnn_drop_rate, use_bn=batch_norm,
                                        use_ln=layer_norm,
                                        item_dnn_hidden_unit=item_dnn_hidden_unit,
                                        item_dnn_activation=item_dnn_activation,
                                        item_dnn_drop_rate=item_dnn_drop_rate, similar_type=similar_type,
                                        l2_v=l2_v, l2_layer=l2_layer)
    result = run_one_model(dataset=dataset, model=model, pre_train_logdir=pre_train_logdir, son_structure_performance_logdir=son_structure_performance_logdir,
                select_structure_type=select_structure_type, alpha=alpha, beta=beta, save_key=save_key,
                evaluate_way=evaluate_way)
    tf.reset_default_graph()












