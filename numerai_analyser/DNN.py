import tensorflow as tf
from tensorflow import keras
from sklearn.base import BaseEstimator
import threading


class DNNVanilla(BaseEstimator):

    lock = threading.Lock()

    def __init__(self, width, depth, activation = 'relu', metrics = ['accuracy']):

        self.width = width
        self.depth = depth
        self.activation = activation
        self.metrics = metrics

        model = keras.Sequential()

        for i in range(depth):
            model.add(keras.layers.Dense(width, activation = activation, kernel_regularizer= keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)))
            model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.Dense(1, activation = 'sigmoid'))

        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
            loss='mse', metrics = metrics)

        self.model = model

    def fit(self, X, y):

        self.model.fit(X.values, y.values, use_multiprocessing=False)

    def predict(self, X):

        output = self.model.predict(X.values).ravel()

        return(output)

    def get_params(self, deep=True):

        return {'width': self.width, 'depth': self.depth, 'activation': self.activation, 'metrics' : self.metrics}





