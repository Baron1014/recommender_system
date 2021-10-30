from scipy import sparse
from deepctr import models
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
from models.evaluation import recall_k
from dataaccessframeworks.data_preprocessing import generate_eval_array
from util.mywandb import WandbLog

class DeepCTRModel:
    def __init__(self):
        self.__models = models
        self.__sparse_features = ['user', 'movie', 'movie_genre', 'user_occupation']
        self.__dense_features = ['user_age']
        self.__target = ['rating']

    def tras_data_to_CTR(self, dataframe):
        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        for feat in self.__sparse_features:
            lbe = LabelEncoder()
            dataframe[feat] = lbe.fit_transform(dataframe[feat])
            
        mms = MinMaxScaler(feature_range=(0, 1))
        dataframe[self.__dense_features] = mms.fit_transform(dataframe[self.__dense_features])
         
        # 2.count #unique features for each sparse field,and record dense feature field name
        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=dataframe[feat].nunique(),embedding_dim=4 )
                           for i,feat in enumerate(self.__sparse_features)] + [DenseFeat(feat, 1,)
                          for feat in self.__dense_features]

        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        return dnn_feature_columns, linear_feature_columns, dataframe

    def FNN(self, dataframe, test_df, test_index, users, items):
        # training 
        dnn_feature_columns, linear_feature_columns, dateframe = self.tras_data_to_CTR(dataframe)
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        # make testing input
        _, _, test_dataframe = self.tras_data_to_CTR(test_df)
        test_model_input = {name:test_dataframe[name] for name in feature_names}
        
        # init evaluation
        log = WandbLog()
        rmse = list()
        recall = list()
        ndcg =list()
        result = dict()
        epochs = 5
        for epoch in range(epochs):
            # 3.generate input data for model
            train, val = train_test_split(dataframe, test_size=0.1, random_state=42)
            train_model_input = {name:train[name] for name in feature_names}
            val_model_input = {name:val[name] for name in feature_names}
            
            model = self.__models.FNN(linear_feature_columns, dnn_feature_columns, task='regression')
            model.compile("adam", "mse",
                    metrics=['mse'], )
            history = model.fit(train_model_input, train[self.__target].values,
                            batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
            pred_ans = model.predict(test_model_input, batch_size=256)

            #result
            real_values = test_dataframe[self.__target].values
            rating_testing_array = generate_eval_array(real_values, test_index, users, items)
            predict_array = generate_eval_array(pred_ans, test_index, users, items)
            rmse.append(mse(pred_ans, real_values, squared=False))
            recall.append(recall_k(rating_testing_array, predict_array))
            ndcg.append(ndcg_score(rating_testing_array, predict_array))
        result['rmse'] = sum(rmse) / len(rmse)
        result['recall@10'] = sum(recall) / len(recall)
        result['ndcg@10'] = sum(ndcg) / len(ndcg)
        log.log_evaluation(result)

        return result