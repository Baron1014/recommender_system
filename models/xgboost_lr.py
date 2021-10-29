# implementation of GBDT+LR
# import relative library

import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import ndcg_score
from dataaccessframeworks.read_data import training_testing_XY
from dataaccessframeworks.data_preprocessing import generate_eval_array
from sklearn.metrics import mean_squared_error as mse
from models.evaluation import recall_k
from util.mywandb import WandbLog

def execute_xgb_lr(X_train, y_train, X_test, y_test, test_index, users, items):
    log = WandbLog()
    rating_testing_array = generate_eval_array(y_test, test_index, users, items)
    rmse = list()
    recall = list()
    ndcg = list()
    result = dict()
    for cross in range(5):
        X_train, X_val, y_train, y_val= training_testing_XY(X_train, y_train)

        print('Start training...')
        # train
        gbm = xgb.XGBRegressor()
        gbm.fit(X_train, y_train)
        print('Start predicting...')
        # predict and get data on leaves, training data
        y_pred = gbm.predict(X_test)
        pred_array = generate_eval_array(y_pred, test_index, users, items)

        # evaluation
        rmse.append(mse(y_test, y_pred))
        recall.append(recall_k(rating_testing_array, pred_array))
        ndcg.append(ndcg_score(rating_testing_array, pred_array))
    
    result["rmse"] = sum(rmse)/len(rmse)
    result["recall"] = sum(recall)/len(recall)
    result["ndcg_score"] = sum(ndcg)/len(ndcg)
    print(result)
    log.log_evaluation(result)

    return result
