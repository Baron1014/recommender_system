import os
import numpy as np
import pandas as pd
from dataaccessframeworks import dataaccesskernel
import configparser
from sklearn.model_selection import train_test_split
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.getcwd()), 'config.ini'))

def get_movielens():

    MovieLens = ['user_movie', 'movie_genre', 'user_age', 'user_occupation']
    movie = dict()

    for i in MovieLens:
        data = dataaccesskernel.read_dat(f'../data/Movielens/{i}.dat')
        print(f"{i}:{data[0:3]}")
        movie[i] = np.array(data)

    return movie

def get_yelp():
    yelp = ['business_category', 'business_city', 'user_business', 'user_compliment']
    d = dict()

    for i in yelp:
        data = dataaccesskernel.read_dat(f'../data/Yelp/{i}.dat')
        print(f"{i}:{data[0:3]}")
        d[i] = np.array(data)

    return d

def get_douban():
    douban = ['book_author', 'book_publisher', 'book_year', 'user_book',
            'user_group', 'user_location']
    d = dict()

    for i in douban:
        data = dataaccesskernel.read_dat(f'../data/Douban/{i}.dat')
        print(f"{i}:{data[0:3]}")
        d[i] = np.array(data)

    return d

def training_testing(data, test_size = float(config['model']['testing_rate'])):
    # 將訓練資料及測試資料切分
    train_data, test_data = train_test_split(data, test_size = test_size)

    return train_data, test_data

def training_testing_XY(X, y, test_size=float(config['model']['testing_rate']), random_state=None):
    # 將訓練資料及測試資料切分
    if random_state:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=random_state)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)

    return X_train, X_test, y_train, y_test

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
