import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dataaccessframeworks.read_data import user_filter

def get_one_hot_feature(data, user_item_col, y_col=2, time_col=3):
    user_items = np.array([list(map(int, data))for data in data[user_item_col]])
    # 使用者評分次數小於三筆則剔除
    filter_data = user_filter(user_items, 0)
    print(filter_data.shape)

    # 取得y
    y = filter_data[:,y_col]
    # 刪除y及時間欄位
    filter_data = np.delete(filter_data, np.s_[y_col:time_col+1], axis=1)
    user, item = user_item_col.split('_')
    # 針對特徵進行map
    for k in data.keys():
        # 取得目前data欄位
        col = len(filter_data[0])
        # 新增加feature vec
        if k==user_item_col:
            continue

        # 處理檔案名稱含有使用者的資料
        elif user in k:
            # 將feature資料轉為map
            user_features = {int(d[0]): int(d[1]) for d in data[k]}
            tmp = list()
            # 將資料對應回去user item
            for i in range(len(filter_data)):
                # 如果有feature則填入
                user_name = filter_data[i][0]
                if user_name in user_features.keys():
                    tmp.append(user_features[user_name])
                # 不存在則補0
                else:
                    tmp.append(0)
            filter_data = np.append(filter_data, np.array(tmp).reshape(-1, 1), axis=1)

        # 處理檔案名稱含有item的資料
        elif item in k:
            # 將feature資料轉為map
            item_features = {int(d[0]): int(d[1]) for d in data[k]}
            tmp = list()
            # 將資料對應回去user item
            for i in range(len(filter_data)):
                # 如果有feature則填入
                item_name = filter_data[i][0]
                if item_name in item_features.keys():
                    tmp.append(item_features[item_name])
                # 不存在則補0
                else:
                    tmp.append(0)
            filter_data = np.append(filter_data, np.array(tmp).reshape(-1, 1), axis=1)

        print(f"col {col} is {k}")


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