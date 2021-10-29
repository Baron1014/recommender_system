import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

import numpy as np
import util.utility as util
from models.collaborative_filtering import get_user_item_matrix
from models.evaluation import recall_k
from sklearn.metrics import ndcg_score
from util.mywandb import WandbLog

def execute_matrix_factorization(users, items, train_data, test_data):
    log = WandbLog()
    # 存放測試資料集的rmse結果
    MF_bias_testing = list()
    # init evaluation
    evaluation = dict()
    user_item = get_user_item_matrix(train_data, users, items)
    test_matrix = get_user_item_matrix(test_data, users, items)

    # init setting global mean
    gu= util.get_u(user_item)
    # init setting user mean as bias
    bu = [util.get_ubias(user_item, i) - gu for i in range(len(users))] 
    # init setting items mean as bias
    bi = [util.get_ibias(user_item, m) - gu for m in range(len(items))] 

    # init lentent vector
    K = int(config["MF"]["latent_vector_number"])
    # init user lentent matrix
    P = np.random.uniform(low=0, high=3, size=(len(users), K))
    # init items lentent matrix
    Q = np.random.uniform(low=0, high=3, size=(len(items), K))

    # parameter
    epochs = int(config["MF"]["epochs"])
    alpha = float(config["MF"]["alpha"])
    l = float(config["MF"]["learning_rate"])

    # 更新次數, init=100
    for epoch in range(epochs):
        # 存放 spuare error 結果
        se_list = list()
        # 針對user有評分過的rating位置進行更新(User Latent Matrix)
        for j in range(len(users)):
            # 找出被使用者j評分過的電影
            movie_index = [i for i, e in enumerate(user_item[j]) if e != 0]
            for m in movie_index:
                # 對u 做偏微分進行ＳＧＤ更新
                tmp_gu = gu - alpha * (((np.dot(P[j], Q[m]) + gu + bu[j] + bi[m]) - user_item[j,m]) + l*(gu))
                # 對bu 做偏微分進行ＳＧＤ更新
                tmp_bu = bu[j] - alpha * (((np.dot(P[j], Q[m]) + gu + bu[j] + bi[m]) - user_item[j,m]) + l*(bu[j]))
                # 對bi 做偏微分進行ＳＧＤ更新
                tmp_bi = bi[m] - alpha * (((np.dot(P[j], Q[m]) + gu + bu[j] + bi[m]) - user_item[j,m]) + l*(bi[m]))
                # 若user item 有值則對Q的相對欄位進行SGD更新, 將更新後user latent matrix先暫存
                tmp = Q[m] - alpha * (((np.dot(P[j], Q[m]) + gu + bu[j] + bi[m]) - user_item[j,m]) * P[j] + l*(Q[m]))
                # 更新 movie latent matrix
                P[j] -= alpha * (((np.dot(P[j], Q[m]) + gu + bu[j] + bi[m]) - user_item[j,m]) * Q[m] + l*(P[j]))
                # 更新 user latent matrix
                Q[m] = tmp
                # 更新bias
                gu = tmp_gu
                bu[j] = tmp_bu
                bi[m] = tmp_bi
                # 計算ＳＥ
                se_list.append(util.se(user_item[j, m], (np.dot(P[j], Q[m]) + gu + bu[j] + bi[m])))
                
        # 進行驗證資料測試
        MF_bias_testing.append(test(test_data, P, Q, gu, bu, bi))
        if epoch % 9 == 0:
            print(f"epoch={epoch}, gu={gu}, bu={np.mean(bu)}, bi={np.mean(bi)}, testing error={MF_bias_testing[-1]}")

    # 各評估指標
    print("start evaluation model...")
    evaluation['rmse']= MF_bias_testing[-1]
    evaluation['recall@10'] = recall_k(test_matrix, np.dot(P, Q.T))
    evaluation['NDCG@10'] = ndcg_score(test_matrix, np.dot(P, Q.T))
    log.log_evaluation(evaluation)
    
    return evaluation

# 進行測試資料驗證評估
def test(test_data, p, q, gu=False, bu=False, bi=False):
    rmse_test = list()

    for test in test_data:
        user = test[0] - 1
        movie = test[1] - 1
        # 判斷是否有bias
        if gu and bu and bi:
            rmse_test.append(util.se(test[2], (np.dot(p[user], q[movie]) + gu + bu[user] + bi[movie])))
        else:
            rmse_test.append(util.se(test[2], (np.dot(p[user], q[movie]))))
    return util.rmse(rmse_test)