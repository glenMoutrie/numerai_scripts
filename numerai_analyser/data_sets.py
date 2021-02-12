import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from .feature_selection import FeatureSelection
from .auto_cluster import ClusterFeature
from abc import ABC, abstractmethod


class DataSet(ABC):
    """

    This module provides the structure for all data set classes.

    The parent class DataSet contains all of the feature mapping that both the training set and test set need to perform.

    """

    full_set = np.ndarray(None)

    poly = None

    def __init__(self, config, data, competition_type, polynomial=False, estimate_clusters=False):

        self.config = config
        self.full_set = data
        self.estimate_clusters = estimate_clusters

        self.all_features = [f for f in list(self.full_set) if "feature" in f]
        self.original_features = self.all_features.copy()
        self.numeric_features = self.all_features

        self.eras = data.era.unique()

        self.category_features = []

        self.competition_type = competition_type
        # self.y_col = 'target_' + competition_type
        self.y_col = 'target'

        self.updateFeaturesList()

        if polynomial:
            self.generatePolynomialFeatures(2, False, False)

        self.N = data.shape[0]

        self.full_index = [i for i in range(0, self.N)]

    def reduceFeatureSpace(self, min_include):

        self.config.logger.info("Reducing Feature Space\nInitial feature set:")
        self.config.logger.info(", ".join(self.numeric_features))

        # If True then this will re-run for every competition
        self.feature_selector = FeatureSelection(self.full_set, self.numeric_features, self.y_col)
        self.numeric_features = self.feature_selector.selectBestFeatures(min_include)

        self.config.logger.info("New feature space:")
        self.config.logger.info(", ".join(self.numeric_features))

    def updateFeaturesList(self, numeric_features=None, category_features=None):

        if numeric_features is None:
            numeric_features = self.numeric_features

        if category_features is None:
            category_features = self.category_features

        self.features = numeric_features + category_features

    def getID(self):
        return self.full_set["id"]

    def getEraIndex(self, era):
        return self.full_set.era == era

    def getX(self, data_type=None, original_features=False, era=None):

        if data_type is None:
            subset = [True] * self.full_set.shape[0]
        else:
            subset = self.full_set["data_type"] == data_type

        if era is not None:
            subset = subset & self.full_set.era == era

        if original_features:
            feature_focus = self.original_features
        else:
            feature_focus = self.features

        return pd.get_dummies(self.full_set.loc[subset, feature_focus])

    def getY(self, data_type=None, era=None):

        if data_type is None:
            subset = np.ones(self.full_set.shape[0], dtype = 'bool')
        else:
            subset = self.full_set["data_type"] == data_type

        if era is not None:
            subset = subset & (self.full_set.era == era)

        return self.full_set.loc[subset, self.y_col]

    def getEras(self, unique_eras=False):

        if unique_eras:
            return self.eras
        else:
            return self.full_set.era

    # TODO: fix poly for full_set
    def generatePolynomialFeatures(self, poly_degree=2, interaction=False, log=True):

        if interaction:

            self.poly = PolynomialFeatures(degree=poly_degree, include_bias=False)

            poly_fit = self.poly.fit_transform(self.full_set[self.numeric_features])

            new_features = self.poly.get_feature_names(self.numeric_features)

            new_features = list(map(lambda x: x.replace(" ", "_"), new_features))

            self.numeric_features = new_features

            self.updateFeaturesList()

            self.full_set = pd.concat([self.full_set["id"],
                                       self.full_set[self.y_col],
                                       self.full_set[self.category_features],
                                       pd.DataFrame(poly_fit, columns=self.numeric_features)], axis=1)

        else:

            new_features = []

            for power in range(2, poly_degree + 1):
                for col in self.numeric_features:
                    feature_name = col + "_" + str(power)
                    self.full_set[feature_name] = np.power(self.full_set[col], power)
                    new_features.append(feature_name)

            if log:

                for col in self.numeric_features:
                    feature_name = "log_" + col
                    self.full_set[feature_name] = np.power(self.full_set[col], power)
                    new_features.append(feature_name)

            self.numeric_features += new_features
            self.features += new_features


class TrainSet(DataSet):
    """
    Train Set

    This class has two key features. One is adding features that are used when testing models,
    a second is the ability to provide

    """

    def __init__(self, config, data, competition_type, polynomial=True, reduce_features=True, test=False,
                 estimate_clusters = False, use_features = None):

        super(TrainSet, self).__init__(config, data, competition_type, polynomial, estimate_clusters)

        if reduce_features and use_features is None:
            if test:
                prob = 0.9
            else:
                prob = 0.01

            self.reduceFeatureSpace(prob)

        elif use_features is not None:

            self.numeric_features = use_features

        self.updateFeaturesList()

        if self.estimate_clusters:

            self.config.logger.info("Estimating Clusters")
            self.cluster_model = ClusterFeature(self.full_set[self.numeric_features], None)

            cluster_id = self.cluster_model.assignClusters(self.full_set[self.numeric_features])

            self.clusters = np.unique(cluster_id)

            self.full_set["cluster"] = pd.Categorical(cluster_id, categories=self.clusters)

            self.features += ["cluster"]

        else:

            self.cluster_model = None

            self.clusters = None

        # A HACK THAT I NEED TO FIX (subset category features to get 0)
        # self.eras = self.full_set[self.category_features].unique()
        # self.full_set[self.category_features] = pd.Categorical(self.full_set[self.category_features],
        #     ordered = True, categories = self.eras)

        # Default value for the split index, this should be updated later externally
        if not (self.full_set.data_type == 'train').all():
            warning_text = 'Not all data_type values have the value "train" in the training set\n'\
            + ", ".join([i + ": " + str(j) for i, j in full_set.data_type.value_counts().to_dict().items()])

            config.logger.warning(warning_text)

    def updateSplit(self, train_ind, test_ind):

        self.full_set.data_type.iloc[test_ind] = 'test'

        self.full_set.data_type.iloc[train_ind] = 'train'


class TestSet(DataSet):
    """
    TestSet

    The Test Set class provides all of the functionality that is unique to the test numerai set.

    A key feature is that any transformations that are made on the train set before modelling must
    also be made on the train set prior to estimation.

    """

    def __init__(self, config, data, competition_type, numeric_features, cluster_model, clusters,
                 polynomial=True, estimate_clusters = False):
        super(TestSet, self).__init__(config, data, competition_type, polynomial=polynomial, estimate_clusters= estimate_clusters)

        self.numeric_features = numeric_features
        self.updateFeaturesList()

        if self.estimate_clusters:
            self.full_set["cluster"] = pd.Categorical(
                cluster_model.assignClusters(self.full_set[self.numeric_features]), categories=clusters)

            self.features += ["cluster"]
            # self.category_features += ["cluster"]

        self.eras = self.full_set.era.unique()


def subsetDataForTesting(data, era_len=100):
    era_len -= 1

    return (
        pd.concat([data.iloc[np.random.shuffle(np.where(data.era == era))][0:era_len] for era in data.era.unique()]))