from lightfm import LightFM
from lightfm.evaluation import recall_at_k
from sklearn.metrics import ndcg_score
from dataaccessframeworks.read_data import training_testing
from dataaccessframeworks.data_preprocessing import generate_eval_array
from sklearn.metrics import mean_squared_error as mse
from util.mywandb import WandbLog

def execute_bpr_fm(train_data, test_data, users, items):
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
        model = LightFM(learning_rate=0.05, loss='bpr')
        model.fit(rating_training_array, epochs=10)
        predict = model.predict(test_data[:,0], test_data[:,1])
        
        # evaluation
        rmse.append(mse(test_data[:,2], predict))
        recall.append(recall_at_k(model, test_data, k=10).mean())
        ndcg.append(ndcg_score(model, test_data, k=10))
    
    result["rmse"] = sum(rmse)/len(rmse)
    result["recall"] = sum(recall)/len(recall)
    result["ndcg_score"] = sum(ndcg)/len(ndcg)
    print(result)
    log.log_evaluation(result)

    return result