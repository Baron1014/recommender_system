from spotlight.evaluation import rmse_score
from spotlight.factorization.explicit import ExplicitFactorizationModel
from sklearn.metrics import ndcg_score
from dataaccessframeworks.read_data import training_testing
from dataaccessframeworks.data_preprocessing import generate_eval_array
from sklearn.metrics import mean_squared_error as mse
from util.mywandb import WandbLog
from models.evaluation import recall_k 


def execute_bpr_mf(train_data, test_data, users, items):
    log = WandbLog()
    rmse = list()
    recall = list()
    ndcg = list()
    result = dict()
    rating_testing_array = generate_eval_array(test_data[:,2], test_data, users, items)
    for cross in range(5):
        train, validation = training_testing(train_data)
        rating_training_array = generate_eval_array(train[:,2], train, users, items)

        # build model
        model = ExplicitFactorizationModel(n_iter=30, loss='bpr')
        model.fit(rating_training_array)

        predict = model.predict(test_data[:,0], test_data[:,1])
        
        # evaluation
        rmse.append(rmse_score(model, test_data))
        recall.append(recall_k(model, test_data))
        ndcg.append(ndcg_score(model, test_data, k=10))
    
    result["rmse"] = sum(rmse)/len(rmse)
    result["recall"] = sum(recall)/len(recall)
    result["ndcg_score"] = sum(ndcg)/len(ndcg)
    print(result)
    log.log_evaluation(result)

    return result