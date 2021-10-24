import os
import wandb
import configparser


class WandbObject:
    def __init__(self):
        self.__config = configparser.ConfigParser()
        self.__config.read(os.path.join(os.path.dirname(__file__), '..', 'config.ini'), encoding='utf-8')

    @staticmethod
    def set_wandb_config(self, params):
        for key, value in params.items():
            if key not in wandb.config._items:
                wandb.config.update({key: value})

class WandbLog:
    def __init__(self):
        self.__config = wandb.config   

    def log_evaluation(self, result):
        # wandb log summary
        wandb.summary['rmse'] = result['rmse']
        wandb.summary['recall@10'] = result['recall@10']
        wandb.summary['NDCG@10'] = result['NDCG@10']
    
    def save_wandb(self, model):
        model.saveModel(wandb.run.dir)