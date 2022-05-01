from scipy import sparse
from deepctr import models
from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
from models.collaborative_filtering import wandb_config
from models.evaluation import recall_k
from dataaccessframeworks.data_preprocessing import generate_eval_array, get_din_data
from util.mywandb import WandbLog

class DeepCTRModel:
    def __init__(self, sparse, dense=None, y=None):
        self.__models = models
        self.__sparse_features = sparse
        self.__dense_features = dense
        self.__target = y
        self.__epochs = 5
        self.__training_epochs = 10
        self.__log = WandbLog()

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


    def tras_data_to_CTR_nodense(self, dataframe):
        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        for feat in self.__sparse_features:
            lbe = LabelEncoder()
            dataframe[feat] = lbe.fit_transform(dataframe[feat])
            
        mms = MinMaxScaler(feature_range=(0, 1))
         
        # 2.count #unique features for each sparse field,and record dense feature field name
        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=dataframe[feat].nunique(),embedding_dim=4 )
                           for i,feat in enumerate(self.__sparse_features)] 
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        return dnn_feature_columns, linear_feature_columns, dataframe
    
    def get_din_xy(self, dataframe, users, items, history, target):
        data_dict, y = get_din_data(dataframe, users, items, watch_history = history, target=target)
        feature_columns = [SparseFeat(feat, vocabulary_size=dataframe[feat].nunique()+1,embedding_dim=8)
                           for i,feat in enumerate(self.__sparse_features)] + [DenseFeat(feat, 1,)
                          for feat in self.__dense_features]
        feature_columns += [
            VarLenSparseFeat(SparseFeat('hist_movie', vocabulary_size=dataframe['movie'].nunique()+1, embedding_dim=4, embedding_name='movie'),
                            maxlen=1682, length_name="seq_length"),
            VarLenSparseFeat(SparseFeat('hist_movie_genre', dataframe['movie_genre'].nunique()+1, embedding_dim=4, embedding_name='movie_genre'), maxlen=1682,
                            length_name="seq_length")]
        # Notice: History behavior sequence feature name must start with "hist_".
        behavior_feature_list = ["movie", "movie_genre"]
        x = {name: data_dict[name] for name in get_feature_names(feature_columns)}

        return x, y, feature_columns, behavior_feature_list

    def FNN(self, dataframe, test_df, test_index, users, items):
        # training 
        dnn_feature_columns, linear_feature_columns, dataframe = self.tras_data_to_CTR(dataframe)
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        # make testing input
        _, _, test_dataframe = self.tras_data_to_CTR(test_df)
        test_model_input = {name:test_dataframe[name] for name in feature_names}
        
        # init evaluation
        rmse = list()
        recall = list()
        ndcg =list()
        result = dict()
        sum_predict = 0
        for epoch in range(self.__epochs):
            print("[{}/{} Cross Validation]".format(epoch, self.__epochs))
            # 3.generate input data for model
            train, val = train_test_split(dataframe, test_size=0.1, random_state=42)
            train_model_input = {name:train[name] for name in feature_names}
            val_model_input = {name:val[name] for name in feature_names}
            
            model = self.__models.FNN(linear_feature_columns, dnn_feature_columns, task='regression')
            model.compile("adam", "mse",
                    metrics=['mse'], )
            model.fit(train_model_input, train[self.__target].values,
                            batch_size=256, epochs=self.__training_epochs, verbose=2, validation_split=0.2, )
            pred_ans = model.predict(test_model_input, batch_size=256)

            #result
            real_values = test_dataframe[self.__target].values
            rating_testing_array = generate_eval_array(real_values, test_index, users, items)
            predict_array = generate_eval_array(pred_ans, test_index, users, items)
            rmse.append(mse(pred_ans, real_values, squared=False))
            recall.append(recall_k(rating_testing_array, predict_array))
            ndcg.append(ndcg_score(rating_testing_array, predict_array))
            sum_predict+= pred_ans
        result['rmse'] = sum(rmse) / len(rmse)
        result['recall@10'] = sum(recall) / len(recall)
        result['ndcg@10'] = sum(ndcg) / len(ndcg)
        self.__log.log_evaluation(result)

        return result, sum_predict/5


    def PNN(self, dataframe, test_df, test_index, users, items, inner=True, outter=True):
        # training 
        dnn_feature_columns, linear_feature_columns, dataframe = self.tras_data_to_CTR(dataframe)
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        # make testing input
        _, _, test_dataframe = self.tras_data_to_CTR(test_df)
        test_model_input = {name:test_dataframe[name] for name in feature_names}
        
        # init evaluation
        rmse = list()
        recall = list()
        ndcg =list()
        result = dict()
        sum_predict = 0
        for epoch in range(self.__epochs):
            print("[{}/{} Cross Validation]".format(epoch, self.__epochs))
            # 3.generate input data for model
            train, val = train_test_split(dataframe, test_size=0.1, random_state=42)
            train_model_input = {name:train[name] for name in feature_names}
            val_model_input = {name:val[name] for name in feature_names}
            
            # model
            model = self.__models.PNN(dnn_feature_columns, task='regression', use_inner=inner, use_outter=outter)
            model.compile("adam", "mse",
                    metrics=['mse'], )
            history = model.fit(train_model_input, train[self.__target].values,
                            batch_size=256, epochs=self.__training_epochs, verbose=2, validation_split=0.2, )
            pred_ans = model.predict(test_model_input, batch_size=256)

            #result
            real_values = test_dataframe[self.__target].values
            rating_testing_array = generate_eval_array(real_values, test_index, users, items)
            predict_array = generate_eval_array(pred_ans, test_index, users, items)
            rmse.append(mse(pred_ans, real_values, squared=False))
            recall.append(recall_k(rating_testing_array, predict_array))
            ndcg.append(ndcg_score(rating_testing_array, predict_array))
            sum_predict += pred_ans
        result['rmse'] = sum(rmse) / len(rmse)
        result['recall@10'] = sum(recall) / len(recall)
        result['ndcg@10'] = sum(ndcg) / len(ndcg)
        self.__log.log_evaluation(result)

        return result, sum_predict / 5


    def CCPM(self, dataframe, test_df, test_index, users, items):
        # training 
        dnn_feature_columns, linear_feature_columns, dataframe = self.tras_data_to_CTR_nodense(dataframe)
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        # make testing input
        _, _, test_dataframe = self.tras_data_to_CTR_nodense(test_df)
        test_model_input = {name:test_dataframe[name] for name in feature_names}
        
        # init evaluation
        rmse = list()
        recall = list()
        ndcg =list()
        result = dict()
        for epoch in range(self.__epochs):
            print("[{}/{} Cross Validation]".format(epoch, self.__epochs))
            # 3.generate input data for model
            train, val = train_test_split(dataframe, test_size=0.1, random_state=42)
            train_model_input = {name:train[name] for name in feature_names}
            val_model_input = {name:val[name] for name in feature_names}
            
            model = self.__models.CCPM(linear_feature_columns, dnn_feature_columns, task='regression')
            model.compile("adam", "mse",
                    metrics=['mse'], )
            history = model.fit(train_model_input, train[self.__target].values,
                            batch_size=256, epochs=self.__training_epochs, verbose=2, validation_split=0.2, )
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
        self.__log.log_evaluation(result)

        return result


    def WD(self, dataframe, test_df, test_index, users, items):
        # training 
        dnn_feature_columns, linear_feature_columns, dataframe = self.tras_data_to_CTR(dataframe)
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        # make testing input
        _, _, test_dataframe = self.tras_data_to_CTR(test_df)
        test_model_input = {name:test_dataframe[name] for name in feature_names}
        
        # init evaluation
        rmse = list()
        recall = list()
        ndcg =list()
        result = dict()
        for epoch in range(self.__epochs):
            print("[{}/{} Cross Validation]".format(epoch, self.__epochs))
            # 3.generate input data for model
            train, val = train_test_split(dataframe, test_size=0.1, random_state=42)
            train_model_input = {name:train[name] for name in feature_names}
            val_model_input = {name:val[name] for name in feature_names}
            
            model = self.__models.WDL(linear_feature_columns, dnn_feature_columns, task='regression')
            model.compile("adam", "mse",
                    metrics=['mse'], )
            history = model.fit(train_model_input, train[self.__target].values,
                            batch_size=256, epochs=self.__training_epochs, verbose=2, validation_split=0.2, )
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
        self.__log.log_evaluation(result)

        return result


    def DCN(self, dataframe, test_df, test_index, users, items):
        # training 
        dnn_feature_columns, linear_feature_columns, dataframe = self.tras_data_to_CTR(dataframe)
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        # make testing input
        _, _, test_dataframe = self.tras_data_to_CTR(test_df)
        test_model_input = {name:test_dataframe[name] for name in feature_names}
        
        # init evaluation
        rmse = list()
        recall = list()
        ndcg =list()
        result = dict()
        for epoch in range(self.__epochs):
            print("[{}/{} Cross Validation]".format(epoch, self.__epochs))
            # 3.generate input data for model
            train, val = train_test_split(dataframe, test_size=0.1, random_state=42)
            train_model_input = {name:train[name] for name in feature_names}
            val_model_input = {name:val[name] for name in feature_names}
            
            model = self.__models.DCN(linear_feature_columns, dnn_feature_columns, task='regression')
            model.compile("adam", "mse",
                    metrics=['mse'], )
            history = model.fit(train_model_input, train[self.__target].values,
                            batch_size=256, epochs=self.__training_epochs, verbose=2, validation_split=0.2, )
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
        self.__log.log_evaluation(result)

        return result


    def NFM(self, dataframe, test_df, test_index, users, items):
        # training 
        dnn_feature_columns, linear_feature_columns, dataframe = self.tras_data_to_CTR(dataframe)
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        # make testing input
        _, _, test_dataframe = self.tras_data_to_CTR(test_df)
        test_model_input = {name:test_dataframe[name] for name in feature_names}
        
        # init evaluation
        rmse = list()
        recall = list()
        ndcg =list()
        result = dict()
        for epoch in range(self.__epochs):
            print("[{}/{} Cross Validation]".format(epoch, self.__epochs))
            # 3.generate input data for model
            train, val = train_test_split(dataframe, test_size=0.1, random_state=42)
            train_model_input = {name:train[name] for name in feature_names}
            val_model_input = {name:val[name] for name in feature_names}
            
            model = self.__models.NFM(linear_feature_columns, dnn_feature_columns, task='regression')
            model.compile("adam", "mse",
                    metrics=['mse'], )
            history = model.fit(train_model_input, train[self.__target].values,
                            batch_size=256, epochs=self.__training_epochs, verbose=2, validation_split=0.2, )
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
        self.__log.log_evaluation(result)

        return result


    def DeepFM(self, dataframe, test_df, test_index, users, items):
        # training 
        dnn_feature_columns, linear_feature_columns, dataframe = self.tras_data_to_CTR(dataframe)
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        # make testing input
        _, _, test_dataframe = self.tras_data_to_CTR(test_df)
        test_model_input = {name:test_dataframe[name] for name in feature_names}
        
        # init evaluation
        rmse = list()
        recall = list()
        ndcg =list()
        result = dict()
        for epoch in range(self.__epochs):
            print("[{}/{} Cross Validation]".format(epoch, self.__epochs))
            # 3.generate input data for model
            train, val = train_test_split(dataframe, test_size=0.1, random_state=42)
            train_model_input = {name:train[name] for name in feature_names}
            val_model_input = {name:val[name] for name in feature_names}
            
            model = self.__models.DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
            model.compile("adam", "mse",
                    metrics=['mse'], )
            history = model.fit(train_model_input, train[self.__target].values,
                            batch_size=256, epochs=self.__training_epochs, verbose=2, validation_split=0.2, )
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
        self.__log.log_evaluation(result)

        return result


    def AFM(self, dataframe, test_df, test_index, users, items):
        # training 
        dnn_feature_columns, linear_feature_columns, dataframe = self.tras_data_to_CTR_nodense(dataframe)
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        # make testing input
        _, _, test_dataframe = self.tras_data_to_CTR_nodense(test_df)
        test_model_input = {name:test_dataframe[name] for name in feature_names}
        
        # init evaluation
        rmse = list()
        recall = list()
        ndcg =list()
        result = dict()
        for epoch in range(self.__epochs):
            print("[{}/{} Cross Validation]".format(epoch, self.__epochs))
            # 3.generate input data for model
            train, val = train_test_split(dataframe, test_size=0.1, random_state=42)
            train_model_input = {name:train[name] for name in feature_names}
            val_model_input = {name:val[name] for name in feature_names}
            
            model = self.__models.AFM(linear_feature_columns, dnn_feature_columns, task='regression')
            model.compile("adam", "mse",
                    metrics=['mse'], )
            model.fit(train_model_input, train[self.__target].values,
                            batch_size=256, epochs=self.__training_epochs, verbose=2, validation_split=0.2, )
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
        self.__log.log_evaluation(result)

        return result


    def xDeepFM(self, dataframe, test_df, test_index, users, items):
        # training 
        dnn_feature_columns, linear_feature_columns, dataframe = self.tras_data_to_CTR(dataframe)
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        # make testing input
        _, _, test_dataframe = self.tras_data_to_CTR(test_df)
        test_model_input = {name:test_dataframe[name] for name in feature_names}
        
        # init evaluation
        rmse = list()
        recall = list()
        ndcg =list()
        result = dict()
        for epoch in range(self.__epochs):
            print("[{}/{} Cross Validation]".format(epoch, self.__epochs))
            # 3.generate input data for model
            train, val = train_test_split(dataframe, test_size=0.1, random_state=42)
            train_model_input = {name:train[name] for name in feature_names}
            val_model_input = {name:val[name] for name in feature_names}
            
            model = self.__models.xDeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
            model.compile("adam", "mse",
                    metrics=['mse'], )
            model.fit(train_model_input, train[self.__target].values,
                            batch_size=256, epochs=self.__training_epochs, verbose=2, validation_split=0.2, )
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
        self.__log.log_evaluation(result)

        return result


    def DIN(self, dataframe, test_dataframe, test_index, users, items, history, target):

        test_x, test_y, _, _ = self.get_din_xy(test_dataframe, users, items, history, target)
        
        # init evaluation
        rmse = list()
        recall = list()
        ndcg =list()
        result = dict()
        for epoch in range(self.__epochs):
            print("[{}/{} Cross Validation]".format(epoch, self.__epochs))
            # 3.generate input data for model
            train, val = train_test_split(dataframe, test_size=0.1, random_state=42)
            train_X, train_y, feature_columns, behavior_feature_list = self.get_din_xy(dataframe, users, items, history, target)

            model = self.__models.DIN(feature_columns, behavior_feature_list, task='regression')
            model.compile("adam", "mse",
                    metrics=['mse'])
            model.fit(train_X, train_y,
                            batch_size=256, epochs=self.__training_epochs, verbose=2, validation_split=0.2, )
            pred_ans = model.predict(test_x, batch_size=256)

            #result
            real_values = test_y
            rating_testing_array = generate_eval_array(real_values, test_index, users, items)
            predict_array = generate_eval_array(pred_ans, test_index, users, items)
            rmse.append(mse(pred_ans, real_values, squared=False))
            recall.append(recall_k(rating_testing_array, predict_array))
            ndcg.append(ndcg_score(rating_testing_array, predict_array))
        result['rmse'] = sum(rmse) / len(rmse)
        result['recall@10'] = sum(recall) / len(recall)
        result['ndcg@10'] = sum(ndcg) / len(ndcg)
        self.__log.log_evaluation(result)

        return result