import tensorflow as tf
from tensorflow import keras
import numpy as np


class DNNVanilla():
	
	def __init__(self, width, depth, activation = 'relu', metrics = ['accuracy']):

		self.width = width
		self.depth = depth
		self.activation = activation

		model = keras.Sequential()

		for i in range(depth):
			model.add(keras.layers.Dense(width, activation = activation))

		model.add(keras.layers.Dense(1, activation = 'sigmoid'))

		model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
			loss='mse', metrics = metrics)

		self.model = model

	def fit(self, X, y):

		self.model.fit(X.values, y.values)

	def predict(self, X):

		output = self.model.predict(X.values).ravel()

		return(output)





