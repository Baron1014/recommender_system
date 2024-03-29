import sys
import os

from scipy import sparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from dataaccessframeworks.read_data import user_filter
from tqdm import tqdm

def generate_with_feature(array, user_feature, item_feature, init_col):
    dataframe = pd.DataFrame(array, columns=init_col)
    tmp = dict()
    for i in range(dataframe.shape[0]):
        # Finding list of unique keys 
        uniqueUserKey = user_feature[1].keys()
        for k in uniqueUserKey:
            if k not in tmp.keys():
                tmp[k] = user_feature[array[i, 0]][k]
            else:
                tmp[k].append(user_feature[array[i, 0]][k][0])
    # 更新user features
    for key in tmp.keys():
        dataframe[key] = tmp[key]

    #items
    tmp = dict()
    for i in range(dataframe.shape[0]):
        # Finding list of unique keys 
        uniqueItmerKey = item_feature[1].keys()
        for k in uniqueItmerKey:
            if k not in tmp.keys():
                tmp[k] = item_feature[array[i, 1]][k]
            else:
                tmp[k].append(item_feature[array[i, 1]][k][0])
    # 更新user features
    for key in tmp.keys():
        dataframe[key] = tmp[key]

    return dataframe

def get_one_hot_feature(data, user_item_col, y_col=2, time_col=3):
    # 取得user及items feature map 
    users_dict, items_dict, features = get_feature_map(data, user_item_col)

    # 將user item 數值轉為integer
    user_items = np.array([list(map(int, data))for data in data[user_item_col]])
    # 使用者評分次數小於三筆則剔除
    filter_data = user_filter(user_items, 0)
    print(filter_data.shape)
    
    # 做特徵的onehot encoding 
    one_hot_encoder_data, y, concat_data = get_onehot_encoding(filter_data[:,:3], users_dict, items_dict, features)

    return one_hot_encoder_data, y, concat_data

# 取得user及items的one hot encoding map
def get_onehot_encoding(data, users_dict, items_dict, features, y_col=2):
    # 取得加上假資料後的data
    concat_data = get_norating_data(data)

    #users_onehot = get_users_onehot(data)
    sparse, dense = get_feature_onehot(concat_data, users_dict, items_dict, features)

    # 取得y
    y = concat_data[:,y_col].reshape(-1,1)

    return np.concatenate((sparse, dense), axis=1), y, concat_data

# 取得加上假資料的部分
def get_norating_data(data):
    # 產生使用者未評分過的items
    norating_dict = generate_norating_data(data)
    # 將為評分資料轉為原始資料型態[user, item, rating]
    norating_data = np.array([[i, norating_dict[i][j], 0]for i in norating_dict.keys() for j in range(len(norating_dict[i]))])
    # concatenate real data & fake data
    concat_data = np.concatenate((data, norating_data), axis=0)

    return concat_data

# 取得使用者one hot
def get_users_onehot(data):
    # init user one hot
    one_hot = dict()
    # 取得user及item個數
    user_number = np.max(data[:,0]) + 1
    # one hot encoding
    user_one_hot = np.eye(user_number)
    users = np.unique(data[:,0])
    for user in users:
        one_hot[user] =  user_one_hot[user]

    return one_hot

# 取得feature one hot
def get_feature_onehot(data, users_feature, items_feature, features_map):
    # 取得user & item個數
    user_number = np.max(data[:,0]) + 1
    item_number = np.max(data[:,1]) + 1
    # one hot encoding
    user_one_hot = np.eye(user_number)[data[:,0]]
    item_one_hot = np.eye(item_number)[data[:,1]]
    sparse = np.concatenate((user_one_hot, item_one_hot), axis=1)
    dense = np.empty((user_one_hot.shape[0], 1), int)

    # create items feature 
    i_feature = items_feature[1].keys()
    for fe in i_feature:
        tmp = list()
        # sparse
        if fe.split("_")[1] != 'year':
            for item in data[:, 1]:
                # 若item沒有該特徵，則捕0
                if item in features_map[fe].keys():
                    item_feature_onehot = features_map[fe][item]
                else:
                    item_feature_onehot = np.zeros(len(tmp[0]))
                tmp.append(item_feature_onehot)
            sparse = np.concatenate((sparse, tmp), axis=1)
        # dense
        else:
            for item in data[:1]:
                item_feature_onehot = features_map[fe][item]
                tmp.append(item_feature_onehot)
            dense = np.concatenate((dense, tmp), axis=1)

    # create user feature
    u_feature = users_feature[1].keys()
    for fe in u_feature:
        tmp = list()
        # sparse
        if fe.split("_")[1] != 'age':
            for user in data[:, 0]:
                user_feature_onehot = features_map[fe][user]
                tmp.append(user_feature_onehot)
            sparse = np.concatenate((sparse, tmp), axis=1)
        # dense
        else:
            for user in data[:, 0]:
                user_feature_onehot = features_map[fe][user]
                tmp.append(user_feature_onehot)
            dense = np.concatenate((dense, tmp), axis=1)

    return sparse, dense


# 隨機產生K筆使用者為評分的item
def generate_norating_data(data, random_K=100):
    users, items = np.unique(data[:,0]), np.unique(data[:,1])
    norating_map = dict()
    for user in users:
        # 取出資料集中所有評分過的items
        rating_items = data[data[:,0]==user][1]
        # 取出使用者尚未評分的items
        norating = np.setdiff1d(items, rating_items)
        # 隨機sample K筆
        norating_map[user] = np.random.choice(norating, random_K)
    
    return norating_map

def generate_feature_onehot_map(target_dict, feature_data, new_feature):
    target_dict[new_feature] = dict()
    # 將特徵資料轉換為數值形式
    feature_data = np.array([list(map(int, data))for data in feature_data])

    # 取得feature數量 + 1
    feature_number = np.max(feature_data[:,1]) + 1
    # one hot encoding
    feature_one_hot = np.eye(feature_number)[feature_data[:,1]]

    # 將feture做onehot
    for i, feature in enumerate(feature_data):
        user = feature[0]
        if user not in target_dict[new_feature].keys():
            target_dict[new_feature][user] = feature_one_hot[i]
        # 若有重複類型則進行one hot的疊加
        else:
            target_dict[new_feature][user] += feature_one_hot[i] 

    return target_dict

# 產生feature  dict
# target為users 或 items
def generate_feature_map(target_dict, feature_data, new_feature):
    # 將feature資料轉為map
    features = {int(d[0]): int(d[1]) for d in feature_data}
    # 將資料對應回去user item
    for key in target_dict.keys():
        # 為目標字典新增一個新特徵值
        if new_feature not in target_dict[key]:
            target_dict[key][new_feature] = list()
        # 如果訓練資料集中未出現該特徵，則補0
        if key not in features.keys():
            feature_value = 0
        else:
            feature_value = features[key]
        target_dict[key][new_feature].append(feature_value)

    return target_dict

    # concate method: filter_data = np.append(filter_data, np.array(tmp).reshape(-1, 1), axis=1)

# 取得 user 及 items 的feature map
def get_feature_map(data, user_item_col):
    user, item = user_item_col.split('_')
    # init user_dict & items dict
    users, items = np.unique(data[user_item_col][:,0]), np.unique(data[user_item_col][:,1])
    users_dict = {int(u):dict() for u in users}
    items_dict = {int(i):dict() for i in items}
    features_onehot = dict()

    for k in data.keys():
        # 新增加feature vec
        if k==user_item_col:
            continue

        # 處理檔案名稱含有使用者的資料
        elif user in k:
            users_dict = generate_feature_map(users_dict, data[k], k)
            features_onehot = generate_feature_onehot_map(features_onehot, data[k], k)
        # 處理檔案名稱含有item的資料
        elif item in k:
            items_dict = generate_feature_map(items_dict, data[k], k)
            features_onehot = generate_feature_onehot_map(features_onehot, data[k], k)

    return users_dict, items_dict, features_onehot

# 產生能夠做recall的矩陣型態
def generate_eval_array(test_values, test_index, users, items):
    output_array = np.zeros((len(users), len(items)))
    for i, tindex in enumerate(test_index):
        # 實際使用者-1 (編號)
        u = tindex[0] - 1
        # 實際item-1 (編號)
        m = tindex[1] - 1
        
        # 存入矩陣相對應位置
        output_array[u][m] = test_values[i]
    
    return output_array
    
# 取得使用者對於電影的評分順序資料
def get_din_data(df, users, items, watch_history, target):
    history = dict()
    output_history = dict()
    for his in watch_history:
        history[his] = np.zeros((len(users), len(items)))
        output_history[f"hist_{his}"] = list()

    for user_i in range(len(users)):
        user = user_i + 1
        for his in watch_history:
            # 取得使用者總共評分的items或items的特徵
            values = df[df['user']==user][his].values
            for i in range(len(values)):
                history[his][user_i, i] = values[i]
    
    # df to dict
    df_dict = dict()
    for col in df.columns:
        if col == target:
            y = df[col].to_numpy()
        else:
            df_dict[col] = df[col].to_numpy()

    # items特徵順序需對應回每個使用者
    for i in tqdm(range(len(df)), desc = "trasfer history items"):
        for dis in watch_history:
            output_history[f"hist_{dis}"].append(history[dis][df.iloc[i, 0] - 1])

    # 轉換成array
    for dis in output_history.keys():
        output_history[dis] = np.array(output_history[dis])

    # 返回合併後的字典
    return {**df_dict, **output_history}, y

