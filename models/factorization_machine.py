import os
import pywFM
import util.utility as util
import configparser
from dataaccessframeworks.read_data import training_testing_XY
config = configparser.ConfigParser()
config.read('config.ini')

LIBFM_PATH = '/Users/baron/libfm/bin/'
os.environ['LIBFM_PATH'] = LIBFM_PATH

def execute_factorization_machine(X, y, X_test, y_test):
    # kfold = 5
    kfold = list()
    result = dict()
    for i in range(5):
        print(f"Start {i} FM Cross-Validation")
        X_train, X_val, y_train, y_val = training_testing_XY(X, y, test_size=float(config["model"]["val_rate"]), random_state=None)

        # reshape y
        y_train = y_train.reshape(1, -1)[0]
        y_test = y_test.reshape(1, -1)[0]

        # define model
        fm = pywFM.FM(task='regression')

        model = fm.run(X_train, y_train, X_test, y_test)
        kfold.append(util.rmse(model.predictions - y_test))

    result['rmse'] = sum(kfold)/len(kfold) 

    return result