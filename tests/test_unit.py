import numerai_analyser.auto_cluster as auto_cluster
from numerai_analyser import NumeraiConfig
from numerai_analyser import NumeraiDataManager
from numerai_analyser import TestType as tt
from numerai_analyser.synthetic_numerai_data import SyntheticNumeraiData

import random
import pandas as pd
import numpy as np
import pytest

"""
Helper functions

The functions below are used to set up various scenarios in the unit tests
"""


def createTestDataManager():
    conf = NumeraiConfig(True, tt.SUBSET_DATA)
    dm = NumeraiDataManager(conf)
    return dm , conf

def getTestDataSets():
    dm, conf = createTestDataManager()
    dm.getCompetitions()
    dm.read(True)
    return dm.getData(dm.getCompetitions()[0], False, False, True)

"""
Unit test classes
"""


class TestAutoCluster:

    test_data = pd.DataFrame({"one": [random.normalvariate(-i,1) for i in range(10)],
        "two": [random.normalvariate(i,1) for i in range(10)],
        "three": [random.normalvariate(i,1) for i in range(10)]})

    new_data = 	pd.DataFrame({"one": [random.normalvariate(-i,1) for i in range(10)],
        "two": [random.normalvariate(i,1) for i in range(10)],
        "three": [random.normalvariate(i,1) for i in range(10)]})


    def test_clustering(self):

        cluster = auto_cluster.ClusterFeature(self.test_data, None)
        cluster.assignClusters(self.new_data)






class TestDataManager():

    dm, conf = createTestDataManager()


    def test_get_competitions(self):

        comps = self.dm.getCompetitions()

        assert type(comps) is list
        assert len(comps) > 0

    # @pytest.mark.slow
    def test_data_manager_synthetic(self):

        self.dm.downloadLatest()
        self.dm.read()

        self.train, self.test = self.dm.getData(self.dm.getCompetitions()[0], False, False, True)





class TestDataSets():

    train, test = getTestDataSets()

    def test_full_set_format(self):

        assert type(self.train.full_set) is pd.DataFrame
        assert type(self.test.full_set) is pd.DataFrame

    def test_getx(self):

        assert self.test.getX().select_dtypes(include= 'number').shape == self.test.getX().shape
        assert self.train.getX().select_dtypes(include='number').shape == self.train.getX().shape

    def test_getx_all_features(self):

        n_features_train = len(self.train.all_features)
        n_features_test = len(self.test.all_features)

        assert n_features_test == n_features_train

        assert np.intersect1d(self.test.getX(all_features = False).columns, self.test.all_features).shape[0] <= n_features_test
        assert np.intersect1d(self.test.getX(all_features = True).columns, self.test.all_features).shape[0] == n_features_test

        assert np.intersect1d(self.train.getX(all_features=False).columns, self.train.all_features).shape[0] <= n_features_test
        assert np.intersect1d(self.test.getX(all_features=True).columns, self.train.all_features).shape[0] == n_features_test

    def test_getx_eras(self):

        for ds in [self.train, self.test]:

            count = 0

            for era_focus in ds.getEras():
                count += ds.getX(era = era_focus).shape[0]

            assert count == ds.full_set[0]






class TestSyntheticData():

    n = random.randint(30, 500)
    m = random.randint(2, 20)
    n_era = random.randint(2, 5)

    comp = ['bernie']

    syn_data = SyntheticNumeraiData(observations=n, num_eras=n_era, features=m, comp=['bernie'])

    def test_synthetic_test_data_size(self):


        assert self.syn_data.getTestData().shape[0] == self.n
        assert self.syn_data.getTestData().shape[1] == (self.m + 4)

    def test_synthetic_train_data_size(self):

        assert self.syn_data.getTrainData().shape[0] == self.n
        assert self.syn_data.getTrainData().shape[1] == (self.m + 4)

    def test_synthetic_labels(self):

        assert ('target_' + self.comp[0]) in self.syn_data.getTestData().columns

    @pytest.mark.skip(reason = "This test often fails, feature needs to be updated")
    def test_era_count(self):
        assert self.syn_data.getTestData().era.unique().size == self.n_era
        assert self.syn_data.getTrainData().era.unique().size == self.n_era

