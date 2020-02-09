import tensorflow as tf
from tensorflow import keras
from numerai_analyser.test_type import TestType
from numerai_analyser.data_manager import *

import numpy as np

if __name__ == "__main__":
	print(tf.__version__)

	dl = DataLoader()

	competitions = dl.getCompetitions()

	dl.downloadLatest()

	dl.read(False, TestType.SYNTHETIC_DATA, 100)

	print(competitions[0])

	train, test = dl.getData(competitions[0], False, False)


	model = keras.Sequential()

	model.add(keras.layers.Dense(10, activation = 'relu'))

	model.add(keras.layers.Dense(10, activation = 'relu'))

	model.add(keras.layers.Dense(10, activation = 'relu'))

	model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
		loss='mse', metrics = ['accuracy'])

	model.fit(train.getX().values, train.getY().values)

	print(model.summary())

	model.predict(test.getX().values)

	model.save('test_model.h5')

