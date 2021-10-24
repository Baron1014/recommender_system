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

def get_yelp():
    yelp = ['business_category', 'business_city', 'user_business', 'user_compliment', 'user_user']
    d = dict()

    for i in yelp:
        data = dataaccesskernel.read_dat(f'../data/Yelp/{i}.dat')
        print(f"{i}:{data[0:3]}")
        d[i] = np.array(data)

    return d

def get_douban():
    douban = ['book_author', 'book_publisher', 'book_year', 'user_book', 'user_user',
            'user_group', 'user_location']
    d = dict()

    for i in douban:
        data = dataaccesskernel.read_dat(f'../data/Douban Book/{i}.dat')
        print(f"{i}:{data[0:3]}")
        d[i] = np.array(data)

    return d

def training_testing(data):
    # 將訓練資料及測試資料切為8:2
    train_data, test_data = train_test_split(data, test_size = float(config['model']['testing_rate']), random_state=int(config['model']['random_state']))

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
