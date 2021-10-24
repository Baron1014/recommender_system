import numpy as np
import pandas as pd
from dataaccessframeworks import dataaccesskernel
import configparser
from sklearn.model_selection import train_test_split
config = configparser.ConfigParser()
config.read('config.ini')

def get_movielens():

    MovieLens = ['user_movie', 'movie_genre', 'movie_movie(knn)', 'user_age', 'user_occupation', 'user_user(knn)']
    movie = dict()

    for i in MovieLens:
        data = dataaccesskernel.read_dat(f'../data/Movielens/{i}.dat')
        print(f"{i}:{data[0:3]}")
        movie[i] = np.array(data)

    return movie

def training_testing(data):
    # 將訓練資料及測試資料切為8:2
    #train_data, test_data = train_test_split(data, test_size = float(config['model']['testing_rate']), random_state=int(config['model']['random_state']))
    train_data, test_data = train_test_split(data, test_size = 0.2, random_state=42)

    return train_data, test_data

# 刪除user column 小於3的使用者資料
def user_filter(data, col):
    df = pd.DataFrame(data)
    # fileter
    df = df.groupby(col).filter(lambda x : len(x)>3)

    return df.to_numpy()



if __name__ == "__main__":
    data = get_movielens()
    f = user_filter(data['user_movie'], 0)
    print(f.shape)
