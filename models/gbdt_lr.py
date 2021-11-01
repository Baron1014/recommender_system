import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import ndcg_score
from dataaccessframeworks.read_data import training_testing_XY
from dataaccessframeworks.data_preprocessing import generate_eval_array
from sklearn.metrics import mean_squared_error as mse
from util.mywandb import WandbLog

def execute_gbdt_lr(X_train, y_train, X_test, y_test):
    log = WandbLog()
    rmse = list()
    recall = list()
    ndcg = list()
    result = dict()
    for cross in range(5):
        X_train, X_val, y_train, y_val= training_testing_XY(X_train, y_train)
        # create dataset for lightgbm
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        params = {
                    'task': 'train',
                    'boosting_type': 'gbdt',
                    'objective': 'binary',
                    'metric': {'binary_logloss'},
                    'num_leaves': 64,
                    'num_trees': 100,
                    'learning_rate': 0.01,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': 0
                }
        # number of leaves,will be used in feature transformation
        num_leaf = 64

        print('Start training...')
        # train
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=100,
                        valid_sets=lgb_train)
        print('Start predicting...')
        # predict and get data on leaves, training data
        y_pred = gbm.predict(X_train, pred_leaf=True)
        print('Writing transformed training data')
        transformed_training_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf],
                                            dtype=np.int64)  # N * num_tress * num_leafs
        for i in range(0, len(y_pred)):
            temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
            transformed_training_matrix[i][temp] += 1
        # test data
        y_pred = gbm.predict(X_test, pred_leaf=True)
        print('Writing transformed testing data')
        transformed_testing_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
        for i in range(0, len(y_pred)):
            temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
            transformed_testing_matrix[i][temp] += 1

        lm = LogisticRegression(penalty='l2',C=0.05) # logestic model construction
        lm.fit(transformed_training_matrix,y_train)  # fitting the data
        y_pred_test = lm.predict_proba(transformed_testing_matrix)   # Give the probabilty on each label
        
        # evaluation
       # rmse.append(mse(test_data[:,2], predict))
       # recall.append(recall_at_k(model, test_data, k=10).mean())
       # ndcg.append(ndcg_score(model, test_data, k=10))
    
    result["rmse"] = sum(rmse)/len(rmse)
    result["recall"] = sum(recall)/len(recall)
    result["ndcg_score"] = sum(ndcg)/len(ndcg)
    print(result)
    log.log_evaluation(result)

    return result
