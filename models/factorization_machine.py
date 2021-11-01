import os
import pywFM
import util.utility as util
import configparser
from util.mywandb import WandbLog
from dataaccessframeworks.read_data import training_testing_XY
from dataaccessframeworks.data_preprocessing import generate_eval_array
from models.evaluation import recall_k 
from sklearn.metrics import ndcg_score
config = configparser.ConfigParser()
config.read('config.ini')

LIBFM_PATH = '/Users/baron/libfm/bin/'
os.environ['LIBFM_PATH'] = LIBFM_PATH

def execute_factorization_machine(X, y, X_test, y_test, test_index, users, items):
    log = WandbLog()
    rating_testing_array = generate_eval_array(y_test, test_index, users, items)
    # kfold = 5
    kfold = list()
    recall = list()
    ndcg = list()
    result = dict()
    sum_predict_values = 0 
    for i in range(5):
        print(f"Start {i} FM Cross-Validation")
        X_train, X_val, y_train, y_val = training_testing_XY(X, y, test_size=float(config["model"]["val_rate"]), random_state=None)

        # reshape y
        y_train = y_train.reshape(1, -1)[0]
        y_test = y_test.reshape(1, -1)[0]

        # define model
        fm = pywFM.FM(task='regression')

        model = fm.run(X_train, y_train, X_val, y_val)
        predict_values = model.predictions
        predict = generate_eval_array(predict_values, test_index, users, items)
        kfold.append(util.rmse(predict_values - y_test))
        recall.append(recall_k(rating_testing_array, predict))
        ndcg.append(ndcg_score(rating_testing_array, predict))
        sum_predict_values += predict_values

    result['rmse'] = sum(kfold)/len(kfold) 
    result['recall@10'] = sum(recall)/len(recall)
    result['NDCG@10'] = sum(ndcg)/len(ndcg)
    log.log_evaluation(result)

    return result, sum_predict_values