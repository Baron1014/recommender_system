from scipy import sparse
from deepctr import models
from deepctr.feature_column import SparseFeat, DenseFeat

class DeepCTRModel:
    def __init__(self):
        self.__models = models 

    def FNN(self, dataframe):
        sparse_features = ['user', 'movie', 'movie_genre', 'user_occupation']
        dense_features = ['user_age']
        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=dataframe[feat].nunique(),embedding_dim=4 )
                           for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                          for feat in dense_features]
        model = self.__models.FNN(sparse_features, dense_features, task='binary')
        model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )
        a = 0