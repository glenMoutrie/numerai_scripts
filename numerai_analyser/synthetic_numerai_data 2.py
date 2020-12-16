import numpy as np
import pandas as pd
import random


"""
SyntheticNumeraiData 

Generates fake data that can be used for unit testing but still adheres to the overall
structure of numerai

"""
class SyntheticNumeraiData():

	def __init__(self, observations = 100, num_eras = 4, features = 10, comp = ['bernie']):

		self.observations = observations

		self.features = ['feature_' + str(i) for i in range(1, features + 1)]

		self.num_eras = num_eras

		self.eras = ['era_' + str(round((i/(self.observations/num_eras)) - 0.5) + 1) for i in range(1, self.observations + 1)]
		self.eras = list(map(lambda x: 'eraX' if x == 'era_1' else x, self.eras))

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

		test_obs = round(self.observations * 0.5)
		valid_obs = round(self.observations * 0.25)
		live_obs = self.observations - test_obs - valid_obs

		test = pd.DataFrame({'id': self.id,
			'era': self.eras,
			'data_type': ['test'] * test_obs + ['validation'] * valid_obs + ['live'] * live_obs})

		for f in self.features:
			test[f] = [random.choice([0,0.25,0.5,0.75,1]) for i in range(0,self.observations)]

		for c in self.comp:
			col_name = 'target_' + c
			test[col_name] = [random.choice([0,0.25,0.5,0.75,1]) for i in range(0, test_obs + valid_obs)] + [None for i in range(0, live_obs)]

		self.test = test


	def getTrainData(self):
		return(self.train)

	def getTestData(self):
		return(self.test)

if __name__ == '__main__':

	test_data = SyntheticNumeraiData()
	print(test_data.getTrainData())
	print(test_data.getTestData())

