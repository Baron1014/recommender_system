import sys
import os

from models.bpr_mf import bpr_mf
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
import numpy as np
import wandb
from dataaccessframeworks.read_data import get_movielens, training_testing, user_filter, training_testing_XY
from dataaccessframeworks.data_preprocessing import get_one_hot_feature, get_norating_data
from models.collaborative_filtering import user_sim_score, item_sim_score
from models.matrix_factorization import execute_matrix_factorization
from models.factorization_machine import execute_factorization_machine
from models.bpr_fm import execute_bpr_fm

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
    #icf(users, movies, training_data, testing_data)
    # 3. Matrix Factorization
    #mf(users, movies, training_data, testing_data)

    # generarte one hot encoding
    # one_hot_x, y, add_fake_data = get_one_hot_feature(data,  'user_movie')
    # X_train, X_test, y_train, y_test = training_testing_XY(one_hot_x, y)
    # _, test_index, _, _ = training_testing_XY(add_fake_data, y)
    # # 4. Factorization Machine
    # fm(X_train, y_train, X_test, y_test, test_index, users, movies)

    # 取得加上使用者未評分的sample假資料
    include_fake = get_norating_data(filter_data[:, :3])
    training_data,  testing_data = training_testing(include_fake)
    # 5. BPR-MF
    bpr_mf(training_data, testing_data, users, movies)
    # 6. BPR-FM
    bpr_fm(training_data, testing_data, users, movies)
    ###################################################################
    ## NN-based RecSys Methods
    ###################################################################
    # 1. FM-supported Neural Networks
    #reuslt = execute_factorization_machine(X_train, y_train, X_test, y_test)

    ###################################################################
    ## Recent NN-based RecSys Methods
    ###################################################################

    ###################################################################
    ## Ensemble Methods
    ###################################################################
def bpr_fm(train_data, test_data, users, movies):
    # init wandb run
    run = wandb.init(project=config['general']['movielens'],
                        entity=config['general']['entity'],
                        group="BPR_FM",
                        reinit=True)
    reuslt = execute_bpr_fm(train_data, test_data, users, movies)
    print(f"FM={reuslt}")
    run.finish()



def fm(X_train, y_train, X_test, y_test, test_index, users, movies):
    # init wandb run
    run = wandb.init(project=config['general']['movielens'],
                        entity=config['general']['entity'],
                        group="FMachine",
                        reinit=True)
    reuslt = execute_factorization_machine(X_train, y_train, X_test, y_test, test_index, users, movies)
    print(f"FM={reuslt}")
    run.finish()

def mf(users, items, training_data, testing_data):
    # init wandb run
    run = wandb.init(project=config['general']['movielens'],
                        entity=config['general']['entity'],
                        group="MF",
                        reinit=True)
    reuslt = execute_matrix_factorization(users, items, training_data, testing_data)
    print(f"MF={reuslt}")
    run.finish()

def ucf(users, items, training_data, testing_data):
    # init wandb run
    run = wandb.init(project=config['general']['movielens'],
                        entity=config['general']['entity'],
                        group="U-CF",
                        reinit=True)
    reuslt = user_sim_score(users, items, training_data, testing_data)
    print(f"UCF={reuslt}")
    run.finish()

def icf(users, items, training_data, testing_data):
    # init wandb run
    run = wandb.init(project=config['general']['movielens'],
                        entity=config['general']['entity'],
                        group="I-CF",
                        reinit=True)
    reuslt = item_sim_score(users, items, training_data, testing_data)
    print(f"ICF={reuslt}")
    run.finish()



if __name__=="__main__":
    main()
