import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dataaccessframeworks.read_data import user_filter

def get_one_hot_feature(data, user_item_col, y_col=2, time_col=3):
    # 取得user及items feature map 
    users_dict, items_dict = get_feature_map(data, user_item_col)

    # 將user item 數值轉為integer
    user_items = np.array([list(map(int, data))for data in data[user_item_col]])
    # 使用者評分次數小於三筆則剔除
    filter_data = user_filter(user_items, 0)
    print(filter_data.shape)

    # 取得y
    y = filter_data[:,y_col].reshape(-1,1)
    # 刪除y及時間欄位
    filter_data = np.delete(filter_data, np.s_[y_col:time_col+1], axis=1)

    # think about how to addition fake data #

    # 取得user及item個數
    user_number = np.max(filter_data[:,0]) + 1
    item_number = np.max(filter_data[:,1]) + 1
    # one hot encoding
    user_one_hot = np.eye(user_number)[filter_data[:,0]]
    item_one_hot = np.eye(item_number)[filter_data[:,1]]
    # concatenate
    ui_one_hot = np.concatenate((user_one_hot, item_one_hot), axis=1)

    # concatenate feature
    for i in range(2, len(filter_data[0])):
        # 取得feature數量 + 1
        feature_number = np.max(filter_data[:,i]) + 1
        # one hot encoding
        feature_one_hot = np.eye(feature_number)[filter_data[:,i]]
        # concatenate
        ui_one_hot = np.concatenate((ui_one_hot, feature_one_hot), axis=1)
    
    print(f"the one hot data's shape is {ui_one_hot.shape}")

    return ui_one_hot, y

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
        target_dict[key][new_feature].append(features[key])

    return target_dict

    # concate method: filter_data = np.append(filter_data, np.array(tmp).reshape(-1, 1), axis=1)

# 取得 user 及 items 的feature map
def get_feature_map(data, user_item_col):
    user, item = user_item_col.split('_')
    # init user_dict & items dict
    users, items = np.unique(data[user_item_col][:,0]), np.unique(data[user_item_col][:,1])
    users_dict = {u:dict() for u in users}
    items_dict = {i:dict() for i in items}

    for k in data.keys():
        # 新增加feature vec
        if k==user_item_col:
            continue

        # 處理檔案名稱含有使用者的資料
        elif user in k:
            users_dict = generate_feature_map(users_dict, data[k], k)
        # 處理檔案名稱含有item的資料
        elif item in k:
            items_dict = generate_feature_map(items_dict, data[k], k)

    return users_dict, items_dict