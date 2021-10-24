import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dataaccessframeworks.read_data import get_movielens, training_testing, user_filter
from models.collaborative_filtering import user_cosine_score

def main():
    # 取得 movielens 資料
    data = get_movielens()
    # str to int
    user_movie = np.array([list(map(int, data))for data in data['user_movie']])
    print(user_movie.shape)
    # 濾除使用者評分小於三筆的資料
    filter_data = user_filter(user_movie, 0)
    print(f"使用者評分大於三次的共有：{filter_data.shape}")
    # 取得電影個數及電影個數
    users, movies = np.unique(filter_data[:,0]), np.unique(filter_data[:,1])
    # 取得訓練資料及測試資料
    training_data,  testing_data = training_testing(filter_data)

    # 計算U-CF
    ucf_reuslt = user_cosine_score(users, movies, training_data, testing_data)
    print(f"UCF={ucf_reuslt}")

if __name__=="__main__":
    main()
