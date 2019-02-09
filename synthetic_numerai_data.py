import numpy as np
import pandas as pd
import random

class SyntheticNumeraiData():

	def __init__(self, observations = 100, num_eras = 4, features = 10, comp = ['bernie']):

		self.observations = observations

		self.features = ['feature_' + str(i) for i in range(1, features + 1)]

		self.num_eras = num_eras

		self.eras = ['era_' + str(round((i/(self.observations/num_eras)) - 0.5) + 1) for i in range(1, self.observations + 1)]

		self.id = ['test_' + str(i) for i in range(1, self.observations + 1)]

		self.comp = comp

		self.generateTrainData()

		self.generateTestData()






	def generateTrainData(self):
		train = pd.DataFrame({'id': self.id,
			'era': self.eras,
			'data_type': 'train'})

		for f in self.features:
			train[f] = [random.normalvariate(0,1) for i in range(0,self.observations)]

		for c in self.comp:
			col_name = 'target_' + c
			train[col_name] = [random.randint(0,1) for i in range(0, self.observations)]

		self.train = train

	def generateTestData(self):

		test = pd.DataFrame({'id': self.id,
			'era': self.eras,
			'data_type': 'test'})

		for f in self.features:
			test[f] = [random.normalvariate(0,1) for i in range(0,self.observations)]

		for c in self.comp:
			col_name = 'target_' + c
			test[col_name] = [None for i in range(0, self.observations)]

		self.test = test


	def getTrainData(self):
		return(self.train)

	def getTestData(self):
		return(self.test)

if __name__ == '__main__':

	test_data = SyntheticNumeraiData()
	print(test_data.getTrainData())
	print(test_data.getTestData())

