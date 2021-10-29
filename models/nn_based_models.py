from deepctr import models

class DeepCTRModel:
    def __init__(self):
        self.__models = models 

    def FNN(self):
        model = self.__models.FNN(['user', 'movie', 'movie_genre', 'user)occupation'], ['user_age'], task='binary')
        model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )
        a = 0