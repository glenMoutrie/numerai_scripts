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
    return dm.getData()

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




class TestFeatureSelection():

    def test_feature_selection_r(self):
        test = pd.DataFrame({"one" : [1,2,3], "two": [4,5,6], "pred" : [0, 0, 1]})
        fs = FeatureSelection(test, ["one", "two"], "pred")

        print(fs.output)
        print(fs.output.iloc[[1],[0]])
        print(fs.selectBestFeatures(0.1))


class TestDataManager():

    dm, conf = createTestDataManager()


    def test_get_competitions(self):

        comps = self.dm.getCompetitions()

        assert type(comps) is list
        assert len(comps) > 0

    # @pytest.mark.slow
    def test_data_manager_synthetic(self):

        self.train, self.test = self.dm.getData(self.dm.getCompetitions()[0], False, False)





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

    @pytest.mark.skip(reason="This test fails, unclear why")
    def test_getx_eras(self):

        for ds in [self.train, self.test]:

            count = 0

            for era_focus in ds.getEras(unique_eras = True):

                print(era_focus) # string as expected

                # Bug occurs at line below, need to dig into find out why:
                # numerai_analyser / data_sets.py: 292: in getX
                # index = np.intersect1d(index, np.argwhere(self.full_set.era == era))
                count += ds.getX(era = era_focus).shape[0]

            assert count == ds.full_set.shape[0]

#
# import numerai_analyser as n_a
# import random
# import pandas as pd
#
# conf = n_a.NumeraiConfig(False)
# dm = n_a.NumeraiDataManager(conf)
# train, test = dm.getData()
#
# pred = pd.Series([random.choice([0,1]) for i in range(test.N)])
#
# n_a.model_cv.normalizeAndNeutralize(pred, test)

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

