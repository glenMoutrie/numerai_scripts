import os
import numerapi as nmapi
import pandas as pd


from .synthetic_numerai_data import SyntheticNumeraiData
from .test_type import TestType
from .data_sets import TestSet, TrainSet


class NumeraiDataManager():
    """

    Numerai Data Manager

    Manages the numerapi and returns the relevant data sets according to the request. Test cases are
    also requested if running a end-to-end test.

    """

    key = None
    api_conn = None
    user = None
    connected = False

    download_loc = None
    pred_file = "predictions.csv"
    comps = None

    data_read = False
    train = None
    test = None

    training_data_file = 'numerai_training_data.csv'
    test_data_file = 'numerai_tournament_data.csv'

    def __init__(self, config):

        self.config = config

        self.user = config.user
        self.key = config.key

        self.download_loc = config.download_loc
        self.connect()


    def getCompetitions(self):

        comps = self.api_conn.get_tournaments()

        comps = [i['name'] for i in comps]

        self.comps = comps

        return(comps)

    def connect(self):

        if self.user is not None and self.key is not None:

            self.api_conn = nmapi.NumerAPI(self.user, self.key)

            self.connected = True

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

        self.config.logger.info("Writing results to " + str(file_name))

        results.to_csv(file_name, index = False)

        self.config.logger.info("Uploading results to Numerai")

        comp_num = self.api_conn.tournament_name2number(name)

        res = self.api_conn.upload_predictions(file_name, tournament=comp_num)
        self.config.logger.info(res)

    def getSubmissionStatus(self):
        print(self.api_conn.submission_status())

    def read(self):

        if self.config.test_run and self.config.test_type is TestType.SYNTHETIC_DATA:

            synthetic_data = SyntheticNumeraiData(comp = self.comps, observations = self.config.test_size)

            self.train = synthetic_data.getTrainData()
            self.test = synthetic_data.getTestData()

        else:

            self.train = pd.read_csv(self.download_loc / self.sub_folder / self.training_data_file, header = 0)
            self.test = pd.read_csv(self.download_loc / self.sub_folder / self.test_data_file, header = 0)

            if self.config.test_run and self.config.test_type is TestType.SUBSET_DATA:

                self.train = subsetDataForTesting(self.train, self.config.test_size)
                self.test = subsetDataForTesting(self.test, self.config.test_size)

    def _getData(self, competition_type, polynomial, reduce_features):

        self.train = TrainSet(config = self.config, data = self.train, 
            competition_type = competition_type, polynomial = polynomial,
            reduce_features = reduce_features, test = self.config.test_run)

        self.test = TestSet(config = self.config, data = self.test, 
            competition_type = competition_type, era_cat = self.train.getEras(unique_eras=True),
            numeric_features = self.train.numeric_features, cluster_model = self.train.cluster_model, 
            clusters = self.train.clusters, polynomial = polynomial)

        return self.train, self.test

    def getData(self, competition_type = None, polynomial = False, reduce_features = False):

        """
        Gets the numerai train and test data sets, either generating the synthetic test data or
        downloading the latest if not already available in the datasets folder

        :param competition_type: A string indicating which competition you using. If none it will default
        to the first competition retrieved from the numerai_api

        :param polynomial: Boolean indicator stating if polynomial features should be used

        :param reduce_features: Boolean indicator stating if you want to reduce the feature space

        :return: A tuple, first is a TrainData object, secod is a TestData object
        """

        if competition_type is None:

            if self.connected:

                if self.comps is None:
                    self.getCompetitions()
                    competition_type = self.comps[0]

                else:

                    competition_type = self.comps[0]

            else:

                competition_type = 'bernie'

        if not self.config.test_run or self.config.test_type is TestType.SUBSET_DATA:
            self.downloadLatest()

        self.read()

        return self._getData(competition_type, polynomial, reduce_features)
        
def subsetDataForTesting(data, era_len = 100):

    era_len -= 1

    return(pd.concat([data.loc[data.era == era][0:era_len] for era in data.era.unique()]))




    