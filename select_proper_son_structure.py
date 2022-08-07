#encoding=utf-8
import math
import os
import pandas as pd
import numpy as np
import json
if __name__=="__main__":
    # dssm_multi_input:  # 1640084  k=40
    # dssm  519944  k=10
    print("way1 select feature & embedding")
    dssm_flop_num = 519944
    change_batch = 1  # 1, 100
    special_dropout = False

    son_structure_flop_dict = np.load('old/son_structure_flop.npy', allow_pickle=True).item()
    epoch_add_select_step = 2

    select_structure_type = 1
    for initial_dropout, special_dropout in [(True, False)]:
        for change_structure_batch in [1]:  # 1, 10, 100
            for drop_keep_rate in [0.85]:
                for random_seed in [2022]:
                    # select_structure_type = 5
                    # for initial_dropout, special_dropout in [(True, False)]:
                    #     for change_structure_batch in [1]:  # 1, 10, 100
                    #         for drop_keep_rate in [0.85]:
                    #             for random_seed in [2022]:
                    #
                    tmp_topK_result = []
                    tmp_topK_performance_flop_result = []

                    son_structure_performance_path = '../../save_model/Movielens/once_for_all/evaluate_son/' \
                                  'son_type_%d_ini%d%d_change_batch_%d_drop_%f_random_%d.npy' % \
                                  (select_structure_type, initial_dropout, special_dropout,
                                   change_structure_batch, drop_keep_rate, random_seed)
                    son_structure_performance_dict = np.load(son_structure_performance_path, allow_pickle=True).item()
                    topK_structure_list = []  # no strict, dssm, 0.75dssm, 0.5dssm, 0.25dssm
                    # strcit_flop_list = [2*dssm_flop_num, dssm_flop_num, dssm_flop_num*0.75,
                    #                     dssm_flop_num*0.5, dssm_flop_num*0.25, 0]
                    strcit_flop_list = [3 * dssm_flop_num, 2 * dssm_flop_num, dssm_flop_num * 1,
                                        dssm_flop_num * 0.75, dssm_flop_num * 0.5, 0]
                    all_son_structure_performance_dict = []
                    for key in son_structure_performance_dict:
                        try:
                            tmp_tuple = (key, sum(son_structure_performance_dict[key]), son_structure_flop_dict[key])
                            all_son_structure_performance_dict.append(tmp_tuple)
                        except Exception as e:
                            print("error:", e)
                    all_son_structure_performance_sorted_list = sorted(all_son_structure_performance_dict,
                                                                       key=lambda x: (-x[1], -x[2], x[0][0], x[0][1]))
                    for strcit_i in range(5):
                        print("strcit_i:", strcit_i)
                        tmp_structure_topK_list = []
                        for structure_i in range(len(all_son_structure_performance_sorted_list)):
                            tmp_structure, tmp_performance, tmp_flop = all_son_structure_performance_sorted_list[structure_i]
                            # print(tmp_flop)
                            assert tmp_flop == son_structure_flop_dict[tmp_structure]
                            assert tmp_performance == sum(son_structure_performance_dict[tmp_structure])
                            if strcit_flop_list[strcit_i] >= son_structure_flop_dict[tmp_structure]:
                                tmp_tuple = (tmp_structure[0], sum(tmp_structure[1:]), son_structure_performance_dict[tmp_structure], son_structure_flop_dict[tmp_structure])
                                # tmp_tuple = (tmp_structure[0], tuple(tmp_structure[1:]), son_structure_performance_dict[tmp_structure], son_structure_flop_dict[tmp_structure])
                                tmp_structure_topK_list.append(tmp_tuple)
                                # tmp_structure_topK_list.append(son_structure_performance_sorted_list[structure_i])
                            if (len(tmp_structure_topK_list) >= 2):
                                break
                        if len(tmp_structure_topK_list) >= 1:
                            topK_structure_list.append(tmp_structure_topK_list)
                        else:
                            print("wrong")
                            continue
                        print(tmp_structure_topK_list)
                        tmp_topK_result += topK_structure_list[-1][0][2]
                        tmp_topK_performance_flop_result.append(tuple((topK_structure_list[-1][0][0], topK_structure_list[-1][0][1])))
                        tmp_topK_performance_flop_result.append(topK_structure_list[-1][0][3])
                        tmp_topK_performance_flop_result += list(np.round(topK_structure_list[-1][0][2], 4))

                    tmp_topK_result = list(np.round(tmp_topK_result, 4))
                    tmp_topK_performance_flop_result_str = ""
                    for tmp_str_i in range(len(tmp_topK_performance_flop_result)):
                        tmp_topK_performance_flop_result_str += str(tmp_topK_performance_flop_result[tmp_str_i]) + ';'
                    print(tmp_topK_performance_flop_result_str)



