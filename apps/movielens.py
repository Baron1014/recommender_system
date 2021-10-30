import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
import numpy as np
import wandb
from dataaccessframeworks.read_data import get_movielens, training_testing, user_filter, training_testing_XY
from dataaccessframeworks.data_preprocessing import get_one_hot_feature, get_norating_data, get_feature_map, generate_with_feature, get_din_data
from models.collaborative_filtering import user_sim_score, item_sim_score
from models.matrix_factorization import execute_matrix_factorization
from models.factorization_machine import execute_factorization_machine
from models.bpr_fm import execute_bpr_fm
from models.bpr_mf import execute_bpr_mf
from models.gbdt_lr import execute_gbdt_lr
from models.xgboost_lr import execute_xgb_lr
from models.nn_based_models import DeepCTRModel

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
    one_hot_x, y, add_fake_data = get_one_hot_feature(data,  'user_movie')
    X_train, X_test, y_train, y_test = training_testing_XY(one_hot_x, y)
    _, test_index, _, _ = training_testing_XY(add_fake_data, y)
    # # 4. Factorization Machine
    # fm(X_train, y_train, X_test, y_test, test_index, users, movies)

    # 取得加上使用者未評分的sample假資料
    include_fake = get_norating_data(filter_data[:, :3])
    training_data,  testing_data = training_testing(include_fake)
    # 5. BPR-MF
    #bpr_mf(training_data, testing_data, users, movies)
    # 6. BPR-FM
    #bpr_fm(training_data, testing_data, users, movies)
    # 7. GBDT + LR
    # gbdt_lr(X_train, y_train, X_test, y_test)
    # 8. xgboost + LR
    #execute_xgb_lr(X_train, y_train, X_test, y_test, test_index, users, movies)
    ###################################################################
    ## NN-based RecSys Methods
    ###################################################################
    # 取得user及items feature map 
    users_dict, items_dict, features = get_feature_map(data, 'user_movie')
    dataframe = generate_with_feature(training_data, users_dict, items_dict, init_col=["user", "movie", "rating"])
    test_dataframe = generate_with_feature(testing_data, users_dict, items_dict, init_col=["user", "movie", "rating"])
    # 1. FM-supported Neural Networks
    #fnn(dataframe, test_dataframe, test_index, users, movies)
    # 2. Product-based Neural Networks
    #ipnn(dataframe, test_dataframe, test_index, users, movies)
    #opnn(dataframe, test_dataframe, test_index, users, movies)
    #pin
    din(dataframe, test_dataframe, test_index, users, movies)
    afm(dataframe, test_dataframe, test_index, users, movies)
    # 3. Convolutional Click Prediction Model 
    ccpm(dataframe, test_dataframe, test_index, users, movies)
    # 4. neumf
    # 5. Wide&Deep
    wd(dataframe, test_dataframe, test_index, users, movies)
    # 6. Deep Drossing
    dcn(dataframe, test_dataframe, test_index, users, movies)
    # 7. Neural Factorization Machine
    nfm(dataframe, test_dataframe, test_index, users, movies)
    # 8. Deep Factorization Machine
    deepfm(dataframe, test_dataframe, test_index, users, movies)


    ###################################################################
    ## Recent NN-based RecSys Methods
    ###################################################################
    # 1. Attentional Factorization Machines
    afm(dataframe, test_dataframe, test_index, users, movies)
    # 3. xDeepFM
    xdeepfm(dataframe, test_dataframe, test_index, users, movies)
    # 4. Deep Interest Network
    din(dataframe, test_dataframe, test_index, users, movies)

    ###################################################################
    ## Ensemble Methods
    ###################################################################
def din(train_df, test_df, test_index, users, movies, watch_history = ['movie', 'movie_genre'], target="rating"):
    run = wandb.init(project=config['general']['movielens'],
                        entity=config['general']['entity'],
                        group="DIN",
                        reinit=True)
    deer = DeepCTRModel(sparse=['user', 'movie', 'movie_genre', 'user_occupation'],
                        dense=['user_age'],
                        y=['rating'])
    result = deer.DIN(train_df, test_df, test_index, users, movies, watch_history, target)
    print(f"DIN={result}")

def xdeepfm(dataframe, testing_data, test_index, users, movies):
    run = wandb.init(project=config['general']['movielens'],
                        entity=config['general']['entity'],
                        group="xDeepFM",
                        reinit=True)
    deer = DeepCTRModel(sparse=['user', 'movie', 'movie_genre', 'user_occupation'],
                        dense=['user_age'],
                        y=['rating'])
    result = deer.xDeepFM(dataframe, testing_data, test_index, users, movies)
    print(f"xDeepFM={result}")

def afm(dataframe, testing_data, test_index, users, movies):
    run = wandb.init(project=config['general']['movielens'],
                        entity=config['general']['entity'],
                        group="AFM",
                        reinit=True)
    # no dense
    deer = DeepCTRModel(sparse=['user', 'movie', 'movie_genre', 'user_occupation', 'user_age'],
                        y=['rating'])
    result = deer.AFM(dataframe, testing_data, test_index, users, movies)
    print(f"AFM={result}")

def deepfm(dataframe, testing_data, test_index, users, movies):
    run = wandb.init(project=config['general']['movielens'],
                        entity=config['general']['entity'],
                        group="DeepFM",
                        reinit=True)
    deer = DeepCTRModel(sparse=['user', 'movie', 'movie_genre', 'user_occupation'],
                        dense=['user_age'],
                        y=['rating'])
    result = deer.DeepFM(dataframe, testing_data, test_index, users, movies)
    print(f"DeepFM={result}")

def nfm(dataframe, testing_data, test_index, users, movies):
    run = wandb.init(project=config['general']['movielens'],
                        entity=config['general']['entity'],
                        group="CCPM",
                        reinit=True)
    deer = DeepCTRModel(sparse=['user', 'movie', 'movie_genre', 'user_occupation'],
                        dense=['user_age'],
                        y=['rating'])
    result = deer.NFM(dataframe, testing_data, test_index, users, movies)
    print(f"NFM={result}")

def dcn(dataframe, testing_data, test_index, users, movies):
    run = wandb.init(project=config['general']['movielens'],
                        entity=config['general']['entity'],
                        group="DCN",
                        reinit=True)
    deer = DeepCTRModel(sparse=['user', 'movie', 'movie_genre', 'user_occupation'],
                        dense=['user_age'],
                        y=['rating'])
    result = deer.DCN(dataframe, testing_data, test_index, users, movies)
    print(f"DCN={result}")

def wd(dataframe, testing_data, test_index, users, movies):
    run = wandb.init(project=config['general']['movielens'],
                        entity=config['general']['entity'],
                        group="W&D",
                        reinit=True)
    deer = DeepCTRModel(sparse=['user', 'movie', 'movie_genre', 'user_occupation'],
                        dense=['user_age'],
                        y=['rating'])
    result = deer.WD(dataframe, testing_data, test_index, users, movies)
    print(f"W&D={result}")

def ccpm(dataframe, testing_data, test_index, users, movies):
    run = wandb.init(project=config['general']['movielens'],
                        entity=config['general']['entity'],
                        group="CCPM",
                        reinit=True)
    # no suppot dense
    deer = DeepCTRModel(sparse=['user', 'movie', 'movie_genre', 'user_occupation', 'user_age'],
                        y=['rating'])
    result = deer.CCPM(dataframe, testing_data, test_index, users, movies)
    print(f"CCPM={result}")

def ipnn(dataframe, testing_data, test_index, users, movies, inner=True, outter=False):
    run = wandb.init(project=config['general']['movielens'],
                        entity=config['general']['entity'],
                        group="IPNN",
                        reinit=True)
    deer = DeepCTRModel(sparse=['user', 'movie', 'movie_genre', 'user_occupation'],
                        dense=['user_age'],
                        y=['rating'])
    result = deer.PNN(dataframe, testing_data, test_index, users, movies, inner=inner, outter=outter)
    print(f"IPNN={result}")

def opnn(dataframe, testing_data, test_index, users, movies, inner=False, outter=True):
    run = wandb.init(project=config['general']['movielens'],
                        entity=config['general']['entity'],
                        group="OPNN",
                        reinit=True)
    deer = DeepCTRModel(sparse=['user', 'movie', 'movie_genre', 'user_occupation'],
                        dense=['user_age'],
                        y=['rating'])
    result = deer.PNN(dataframe, testing_data, test_index, users, movies, inner=inner, outter=outter)
    print(f"OPNN={result}")

def pin(dataframe, testing_data, test_index, users, movies, inner=True, outter=True):
    run = wandb.init(project=config['general']['movielens'],
                        entity=config['general']['entity'],
                        group="PIN",
                        reinit=True)
    deer = DeepCTRModel(sparse=['user', 'movie', 'movie_genre', 'user_occupation'],
                        dense=['user_age'],
                        y=['rating'])
    result = deer.PNN(dataframe, testing_data, test_index, users, movies, inner=inner, outter=outter)
    print(f"PIN={result}")

def fnn(dataframe, testing_data, test_index, users, movies):
    run = wandb.init(project=config['general']['movielens'],
                        entity=config['general']['entity'],
                        group="FNN",
                        reinit=True)
    deer = DeepCTRModel(sparse=['user', 'movie', 'movie_genre', 'user_occupation'],
                        dense=['user_age'],
                        y=['rating'])
    result = deer.FNN(dataframe, testing_data, test_index, users, movies)
    print(f"FNN={result}")

    run.finish()

def xgb_lr(X_train, y_train, X_test, y_test, test_index, users, items):
    # init wandb run
    run = wandb.init(project=config['general']['movielens'],
                        entity=config['general']['entity'],
                        group="XGB_LR",
                        reinit=True)
    reuslt = execute_xgb_lr(X_train, y_train, X_test, y_test, test_index, users, items)
    print(f"XGB_LR={reuslt}")
    run.finish()


def gbdt_lr(X_train, y_train, X_test, y_test):
    # init wandb run
    run = wandb.init(project=config['general']['movielens'],
                        entity=config['general']['entity'],
                        group="GBDT_LR",
                        reinit=True)
    reuslt = execute_gbdt_lr(X_train, y_train, X_test, y_test)
    print(f"GBDT_LR={reuslt}")
    run.finish()

def bpr_mf(train_data, test_data, users, movies):
    # init wandb run
    run = wandb.init(project=config['general']['movielens'],
                        entity=config['general']['entity'],
                        group="BPR_MF",
                        reinit=True)
    reuslt = execute_bpr_mf(train_data, test_data, users, movies)
    print(f"BPR_MF={reuslt}")
    run.finish()


def bpr_fm(train_data, test_data, users, movies):
    # init wandb run
    run = wandb.init(project=config['general']['movielens'],
                        entity=config['general']['entity'],
                        group="BPR_FM",
                        reinit=True)
    reuslt = execute_bpr_fm(train_data, test_data, users, movies)
    print(f"BPR_FM={reuslt}")
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
