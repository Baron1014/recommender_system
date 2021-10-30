from scipy import sparse
from deepctr import models
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score

class DeepCTRModel:
    def __init__(self):
        self.__models = models 

    def FNN(self, dataframe):
        sparse_features = ['user', 'movie', 'movie_genre', 'user_occupation']
        dense_features = ['user_age']
        target = ['rating']
        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        for feat in sparse_features:
            lbe = LabelEncoder()
            dataframe[feat] = lbe.fit_transform(dataframe[feat])
        mms = MinMaxScaler(feature_range=(0, 1))
        dataframe[dense_features] = mms.fit_transform(dataframe[dense_features])
        
        # 2.count #unique features for each sparse field,and record dense feature field name
        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=dataframe[feat].nunique(),embedding_dim=4 )
                           for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                          for feat in dense_features]

        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        # 3.generate input data for model

        train, test = train_test_split(dataframe, test_size=0.1, random_state=42)
        train_model_input = {name:train[name] for name in feature_names}
        test_model_input = {name:test[name] for name in feature_names}
        
        model = self.__models.FNN(linear_feature_columns, dnn_feature_columns, task='regression')
        model.compile("adam", "mse",
                  metrics=['mse'], )
        history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
        pred_ans = model.predict(test_model_input, batch_size=256)
