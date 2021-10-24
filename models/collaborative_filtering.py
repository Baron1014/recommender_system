import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import heapq
import copy
import numpy as np
import util.utility as util
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
from tqdm import tqdm
from models.evaluation import recall_k
from sklearn.metrics import ndcg_score
from util.mywandb import WandbObject, WandbLog

# 將資料轉換為矩陣形式
def get_user_item_matrix(train_data, users, items):
    # init user_matrix as zero matrix
    user_matrix = np.zeros((len(users), len(items)))

    for i in tqdm(range(len(users)), desc='data transfer user matrix'):
        '''
        train_data[train_data[:,0] == u] : 過濾出u使用者所有的評分資料
        train_data[train_data[:,0] == u][:,1]: 取得u使用者所有評分過的項目名稱
        '''
        rate_index = train_data[train_data[:,0] == users[i]][:,1]
        for rate in rate_index:
            '''
            user_matrix[u-1, rate-1]: 欲設置的rateing位置
            train_data[(train_data[:,0] == u) & (train_data[:,1] == rate)]: 取出u使用者對於評論過特定項目的資料
            '''
            user_matrix[i, rate-1] = train_data[(train_data[:,0] == users[i]) & (train_data[:,1] == rate)][:,2].item()

    return user_matrix


def user_sim_score(users, items, train_data, test_data):
    log = WandbLog()
    wandb_config()
    # 取得資料矩陣
    user_matrix = get_user_item_matrix(train_data, users, items)
    test_matrix = get_user_item_matrix(test_data, users, items)

    # 計算bias
    bias_matrix = util.get_bias(user_matrix, users, items)
    # 計算cosine
    cosine, pcc = util.get_sim(users, user_matrix)
    sim = {"cos":cosine, "pcc":pcc}

    # 取出前K個相似度最大的使用者名稱，並計算square error
    k = int(config['CF']['user_K'])
    evaluation = dict()
    for s in sim.keys():
        delta_list = list()
        result = dict()
        for i in tqdm(sim[s].keys(), desc=f"UCF predicting {s} score with {k}"):
            # Suv: 取出前K個最相似的使用者相似度 ex:K=3, output=[0.378, 0.353, 0.336]
            Suv = heapq.nlargest(k ,sim[s][i])
            # top_sim_index: 取出與使用者i最為相似的前K個使用者 ex:K=3, output=[915, 406, 214]
            top_sim_index = list(map(sim[s][i].index, heapq.nlargest(k,sim[s][i])))
            # recall
            prediction = list()
            # 計算相似使用者與使用者i的評分誤差
            for m in range(len(items)):
                # 取得使用者i的評分(ground truth)
                rth = test_matrix[i, m]
                # 如果使用者i有進行評分，則才納入計算RMSE
                if rth != 0:
                    # 之後需剔除對電影m未評分的相似使用者，因此先進行複製，才不會影響下一部電影的計算
                    copy_Suv = copy.deepcopy(Suv)
                    # R: 若相似使用者對電影 m 有評分則進行調整
                    R = list()
                    # 判斷相似使用者是否對電影ｍ有評分，若有評分則將原始評分減去該使用者對電影m的bias
                    for c, j in enumerate(top_sim_index):
                        if  test_matrix[j, m] == 0:
                            R.append(0)
                            copy_Suv[c] = 0
                        else:
                            R.append(test_matrix[j, m] - bias_matrix[j, m])
                    # 如果所有相似使用者都沒評分則跳過此次計算
                    if sum(R) != 0:
                        # 預測使用者i對於第m部電影的評分 + 使用者i對電影m的偏差
                        Rui = predict(copy_Suv, R) + bias_matrix[i, m]
                        # 計算square error
                        delta_list.append(util.se(rth, Rui))
                        # 儲存預測結果, 並取四捨五入
                        prediction.append(round(Rui))
                    else:
                        prediction.append(0)
                else:
                    prediction.append(0)
            # 儲存所有使用者預測結果
            result[i] = prediction

        # 各評估指標
        evaluation[f'{s}_rmse']= util.rmse(delta_list)
        evaluation[f'{s}_recall@10'] = recall_k(test_matrix, result) 
        evaluation[f'{s}_NDCG@10']=ndcg_score(test_matrix, np.array([result[l] for l in result.keys()]), k=10)

    log.log_evaluation(evaluation)
    return evaluation

def item_sim_score(users, items, train_data, test_data):
    log = WandbLog()
    wandb_config()
    # 取得資料矩陣
    user_matrix = get_user_item_matrix(train_data, users, items)
    test_matrix = get_user_item_matrix(test_data, users, items)
    item_matrix = user_matrix.T 
    item_test = test_matrix.T 

    # 計算bias
    bias_matrix = util.get_bias(user_matrix, users, items)
    item_bias = bias_matrix.T

    # 計算sim
    cosine, pcc = util.get_sim(items, item_bias)
    sim = {"cos":cosine, "pcc":pcc}

    # 取出前K個相似度最大的使用者名稱，並計算square error
    k = int(config['CF']['user_K'])
    evaluation = dict()
    for s in sim.keys():
        delta_list = list()
        result = dict()
        for i in tqdm(sim[s].keys(), desc=f"ICF predicting {s} score with {k}"):
            # Siv: 取出前K個最相似的電影相似度 ex:K=3, output=[0.378, 0.353, 0.336]
            Siv = heapq.nlargest(k,sim[s][i])
            # top_sim_index: 取出與電影i最為相似的前K個電影 ex:K=3, output=[915, 406, 214]
            top_sim_index = list(map(sim[s][i].index, heapq.nlargest(k,sim[s][i])))
            # recall
            prediction = list()
            # 計算相似電影與電影i的評分誤差
            for u in range(len(users)):
                # 取得使用者u的評分(ground truth)
                rth = item_test[i, u]
                # 如果使用者u有評分，則才納入計算RMSE
                if rth != 0:
                    # 之後需剔除對電影i未評分的相似電影，因此先進行複製，才不會影響下一位使用者的計算
                    copy_Suv = copy.deepcopy(Siv)
                    # R: 若相似電影有被使用者u評分則進行調整
                    R = list()
                    # 判斷相似電影是否有被使用者u評分，若有評分則將原始評分減去使用者u對該電影的bias
                    for c, j in enumerate(top_sim_index):
                        if  item_test[j, u] == 0:
                            R.append(0)
                            copy_Suv[c] = 0
                        else:
                            R.append(item_test[j, u] - item_bias[j, u])
                    # 如果所有相似使用者都沒評分則跳過此次計算
                    if sum(R) != 0:
                        # 預測使用者u對於第i部電影的評分 + 使用者u對電影i的偏差
                        Rui = predict(copy_Suv, R) + item_bias[i, u]
                        # 計算square error
                        delta_list.append(util.se(rth, Rui))
                        # 儲存預測結果, 並取四捨五入
                        prediction.append(round(Rui))
                    else:
                        prediction.append(0)
                else:
                    prediction.append(0)
            # 儲存所有使用者預測結果
            result[i] = prediction

        # 各評估指標
        evaluation[f'{s}_rmse']= util.rmse(delta_list)
        evaluation[f'{s}_recall@10'] = recall_k(item_test, result) 
        evaluation[f'{s}_NDCG@10']=ndcg_score(item_test, np.array([result[l] for l in result.keys()]), k=10)

    log.log_evaluation(evaluation)
    return evaluation

# 推測評分
def predict(S, R):
    s = np.sum(S)
    if s == 0:
        return 0

    return np.dot(S,R)/ s

def wandb_config():
    wandb_object = WandbObject()
    wandb_object.set_wandb_config({"testing_rate": float(config['model']['testing_rate'])})
    wandb_object.set_wandb_config({"random_state": int(config['model']['random_state'])})
    wandb_object.set_wandb_config({"similar_user_K": int(config['CF']['user_K'])})
    wandb_object.set_wandb_config({"similar_item_K": int(config['CF']['item_K'])})