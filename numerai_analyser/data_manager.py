import io
import os
import numerapi as nmapi
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from .synthetic_numerai_data import SyntheticNumeraiData
from .test_type import TestType
from .data_sets import TestSet, TrainSet

"""

Numerai Data Manager

Manages the numerapi and returns the relevant data sets according to the request. Test cases are
also requested if running a end-to-end test.

"""
class NumeraiDataManager():

    key = None
    api_conn = None
    user = ""
    download_loc = ""
    pred_file = "predictions.csv"
    comps = None

    training_data_file = 'numerai_training_data.csv'
    test_data_file = 'numerai_tournament_data.csv'

    def __init__(self, config):

        self.config = config

        self.readKey(config.key_loc)
        self.download_loc = config.download_loc
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

        self.sub_folder = "numerai_dataset_" + str(round_num)

        if not self.sub_folder in os.listdir(self.download_loc):
            self.api_conn.download_current_dataset(self.download_loc, unzip = True)
        else:
            self.config.logger.info("Competion data for round " + str(round_num) + " already downloaded.")

        self.sub_folder = self.sub_folder

    def uploadResults(self, results, name):

        file_name = self.download_loc / self.sub_folder / (self.config.time_file_safe + "_" + self.pred_file)

        self.config.logger.info("Writing results to " + file_name)

        results.to_csv(file_name, index = False)

        self.config.logger.info("Uploading results to Numerai")

        comp_num = self.api_conn.tournament_name2number(name)

        res = self.api_conn.upload_predictions(file_name, tournament=comp_num)
        self.config.logger.info(res)

    def getSubmissionStatus(self):
        print(self.api_conn.submission_status())

    def read(self, test = False, test_type = TestType.SYNTHETIC_DATA, subset_size = 100):

        if test and test_type is TestType.SYNTHETIC_DATA:

            synthetic_data = SyntheticNumeraiData(comp = self.comps, observations = subset_size)

            self.train = synthetic_data.getTrainData()
            self.test = synthetic_data.getTestData()

        else:

            self.train = pd.read_csv(self.download_loc / self.sub_folder / self.training_data_file, header = 0)
            self.test = pd.read_csv(self.download_loc / self.sub_folder / self.test_data_file, header = 0)

            if test_type is TestType.SUBSET_DATA:

                self.train = subsetDataForTesting(self.train, subset_size)
                self.test = subsetDataForTesting(self.test, subset_size)

    def getData(self, competition_type, polynomial, reduce_features, test):

        self.train = TrainSet(config = self.config, data = self.train, 
            competition_type = competition_type, polynomial = polynomial,
            reduce_features = reduce_features, test = test)

        self.test = TestSet(config = self.config, data = self.test, 
            competition_type = competition_type, era_cat = self.train.getEras(), 
            numeric_features = self.train.numeric_features, cluster_model = self.train.cluster_model, 
            clusters = self.train.clusters, polynomial = polynomial)

        return self.train, self.test
        
def subsetDataForTesting(data, era_len = 100):

    era_len -= 1

    return(pd.concat([data.loc[data.era == era][0:era_len] for era in data.era.unique()]))

if __name__ == "__main__":

    from .config import NumeraiConfig

    conf = NumeraiConfig(True, TestType.SUBSET_DATA)
    dl = NumeraiDataManager(conf)

    dl.downloadLatest()
    dl.read()
    dl.uploadResults(dl.getCompetitions()[0])



    