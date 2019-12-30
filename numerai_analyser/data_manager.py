import io
import os
import numerapi as nmapi
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from .data_sets import *
from .synthetic_numerai_data import SyntheticNumeraiData

class NumeraiDataManager():

    key = None
    api_conn = None
    user = ""
    download_loc = ""
    pred_file = "predictions.csv"
    comps = None

    def __init__(self, key_loc = "api_key", download_loc = "datasets/"):

        self.readKey(key_loc)
        self.download_loc = download_loc
        self.connect()


    def getCompetitions(self):

        comps = self.api_conn.get_tournaments()

        comps = [i['name'] for i in comps]

        self.comps = comps

        return(comps)


    def readKey(self, key_loc):

        file = io.open(key_loc)

        output = file.read()
        output = output.split("\n")

        self.user = output[0]
        self.key = output[1]

        file.close()

    def connect(self):

        self.api_conn = nmapi.NumerAPI(self.user, self.key)

    def downloadLatest(self):

        round_num = self.api_conn.get_current_round()
        # Hack as api appears to have broken...
        # round_num = 148

        self.sub_folder = "numerai_dataset_" + str(round_num)

        if not self.sub_folder in os.listdir(self.download_loc):
            self.api_conn.download_current_dataset(self.download_loc, unzip = True)
        else:
            print("Competion data for round " + str(round_num) + " already downloaded.")

        self.sub_folder = self.sub_folder + "/"

        # The api put the folder in a sub dir. Solustion left here, need to check on this later...
        if round_num == 131:
            self.sub_folder += "numerai_datasets/"

    def uploadResults(self, name):

        comp_num = self.api_conn.tournament_name2number(name)
        res = self.api_conn.upload_predictions(self.download_loc + self.sub_folder + self.pred_file, tournament=comp_num)
        print(res)

    def getSubmissionStatus(self):
        print(self.api_conn.submission_status())




class DataLoader(NumeraiDataManager):

    training_data_file = 'numerai_training_data.csv'
    test_data_file = 'numerai_tournament_data.csv'


    def read(self, test = False):

        if test:
            synthetic_data = SyntheticNumeraiData(comp = self.comps)

            self.train = synthetic_data.getTrainData()
            self.test = synthetic_data.getTestData()

        else:

            self.train = pd.read_csv(self.download_loc + self.sub_folder + self.training_data_file, header = 0)
            self.test = pd.read_csv(self.download_loc + self.sub_folder + self.test_data_file, header = 0)


        
    def write(self, output):
        output.to_csv(self.download_loc + self.sub_folder + self.pred_file, index = False)

    def getData(self, competition_type):
        self.train = TrainSet(self.train, competition_type)
        self.test = TestSet(self.test, competition_type, self.train.getEras(), self.train.numeric_features, self.train.cluster_model,  self.train.clusters)

        return self.train, self.test
        


if __name__ == "__main__":
    dl = DataLoader()
    dl.downloadLatest()
    dl.uploadResults('bernie')



    