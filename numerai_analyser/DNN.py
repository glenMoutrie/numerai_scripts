import tensorflow as tf
from tensorflow import keras
from sklearn.base import BaseEstimator
import threading


class DNNVanilla(BaseEstimator):

    lock = threading.Lock()

    def __init__(self, width=10, depth=10, activation = 'relu', metrics = ['accuracy'], epochs = 1):

        self.width = width
        self.depth = depth
        self.activation = activation
        self.metrics = metrics
        self.epochs = epochs

        self.compile_param = {'optimizer': tf.keras.optimizers.Adam(0.001),
                              'loss': 'mse',
                              'metrics' : metrics }

        model = keras.Sequential()

        for i in range(depth):
            model.add(keras.layers.Dense(width, activation = activation, kernel_regularizer= keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)))
            model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.Dense(1, activation = 'sigmoid'))

        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
            loss='mse', metrics = metrics)

        self.model = model

    def fit(self, X, y):

        self.model.fit(X.values, y.values, use_multiprocessing=False, epochs = self.epochs)

    def predict(self, X):

        output = self.model.predict(X.values).ravel()

        return(output)

    def to_json(self):

        return self.model.to_json()

    def from_json(self, json):

        self.model = keras.models.model_from_json(json)
        self.model.compile(**self.compile_param)






