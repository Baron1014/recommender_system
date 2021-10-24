import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
import numpy as np
import wandb
from dataaccessframeworks.read_data import get_movielens, training_testing, user_filter
from models.collaborative_filtering import user_sim_score, item_sim_score

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
    ###################################################################
    ## Typical RecSys Methods
    ###################################################################
    # 1. U-CF-cos & U-CF-pcc
    #ucf(users, movies, training_data, testing_data)
    # 2. I-CF-cos & I-CF-pcc
    icf(users, movies, training_data, testing_data)


    ###################################################################
    ## NN-based RecSys Methods
    ###################################################################

    ###################################################################
    ## Recent NN-based RecSys Methods
    ###################################################################

    ###################################################################
    ## Ensemble Methods
    ###################################################################


def ucf(users, movies, training_data, testing_data):
    ##### 計算U-CF
    # init wandb run
    run = wandb.init(project=config['general']['movielens'],
                        entity=config['general']['entity'],
                        group="U-CF",
                        reinit=True)
    reuslt = user_sim_score(users, movies, training_data, testing_data)
    print(f"UCF={reuslt}")
    run.finish()

def icf(users, movies, training_data, testing_data):
    #### 計算I-CF
    # init wandb run
    run = wandb.init(project=config['general']['movielens'],
                        entity=config['general']['entity'],
                        group="I-CF",
                        reinit=True)
    reuslt = item_sim_score(users, movies, training_data, testing_data)
    print(f"ICF={reuslt}")
    run.finish()



if __name__=="__main__":
    main()
