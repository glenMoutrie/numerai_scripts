import io
import os
import numerapi as nmapi
import pandas as pd
import numpy as np

class NumeraiDataManager():

	key = None
	api_conn = None
	user = ""
	download_loc = ""

	def __init__(self, key_loc = "api_key", user = "GBOT", download_loc = "datasets"):

		self.user = user
		self.readKey(key_loc)
		self.connect()


	def readKey(self, key_loc):

		file = io.open(key_loc)


		self.key = file.read()

		file.close()

	def connect(self):

		self.api_conn = nmapi.NumerAPI(self.user, self.key)

	def downloadLatest(self):

		round_num = api.get_current_round()
		file_name = "numerai_dataset_" + str(round_num)

		if file_name in os.listdir(self.download_loc):
			self.api_conn.download_current_dataset(self.download_loc + "/", unzip = True)
		else:
			print("Competion data for round " + round_num + " already downloaded.")

	def uploadResults():
		pass




class DataLoader(NumeraiDataManager):

    training_data_file = 'numerai_training_data.csv'
    test_data_file = 'numerai_tournament_data.csv'


    def read(self):
        self.train = pd.read_csv(self.download_loc + self.training_data_file, header = 0)
        self.test = pd.read_csv(self.download_loc + self.test_data_file, header = 0)

        self.features = [f for f in list(self.train) if "feature" in f]

    def write(self, output):
        output.to_csv(self.download_loc + "predictions.csv", index = False)

    def getData(self, competition_type):
        self.train = DataSet(self.train[self.features], self.train['target_' + competition_type])
        self.test = self.test

        return self.train, self.test
		




class DataSet():

    y_full = np.ndarray(None)
    x_full = np.ndarray(None)

    y_test_split = np.ndarray(None)
    y_train_split = np.ndarray(None)

    x_test_split = np.ndarray(None)
    x_train_split = np.ndarray(None)

    

    def __init__(self, X, Y):

        self.y_full = self.y_test_split = Y
        self.x_full = self.x_test_split = X

        self.N = Y.shape[0]
        self.split_index = {'train' : [i for i in range(0,self.N)], 'test' : []}

    def updateSplit(self, train_ind, test_ind):

        self.y_train_split = self.y_full[train_ind]
        self.x_train_split = self.x_full.iloc[train_ind]

        self.x_test_split = self.x_full.iloc[test_ind]
        self.y_test_split = self.y_full[test_ind]

        self.split_index = {'train' : train_ind, 'test' : test_ind}

    def getTrainingData(self):
        return self.y_train_split, self.x_train_split

    def getTestData(self):
        return self.y_test_split, self.x_test_split

    def getY(self, train):
        if train:
            return self.y_train_split
        else:
            return self.y_test_split

    def getX(self, train):
        if train:
            return self.x_train_split
        else:
            return self.x_test_split

    def getXFull(self):
        return self.x_full

    def getYFull(self):
        return self.y_full


if __name__ == "__main__":
	dl = DataLoader()



    