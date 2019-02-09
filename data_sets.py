import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from feature_selection import FeatureSelection
from auto_cluster import ClusterFeature

class DataSet():

    full_set = np.ndarray(None)

    poly = None

    test = False

    def __init__(self, data, competition_type):

        self.full_set = data

        # self.test = True

        self.numeric_features = [f for f in list(self.full_set) if "feature" in f]

        # Test just the first four features
        if self.test:
            self.numeric_features = self.numeric_features[0:3]
        
        self.category_features = "era"

        self.competition_type = competition_type
        self.y_col = 'target_' + competition_type

        self.updateFeaturesList()

        self.generatePolynomialFeatures(2, False)

        if self.test:
            print(self.full_set)

        self.N = data.shape[0]

    def reduceFeatureSpace(self, min_include):

        print("Reducing Feature Space\nInitial feature set:")
        print(self.numeric_features)

        # If True then this will re-run for every competion
        self.feature_selector = FeatureSelection(self.full_set, self.numeric_features, self.y_col)
        self.numeric_features = self.feature_selector.selectBestFeatures(min_include)

        print("New feature space:")
        print(self.numeric_features)


    def updateFeaturesList(self, numeric_features = None, category_features = None):

        if numeric_features is None:
            numeric_features = self.numeric_features

        if category_features is None:
            category_features = self.category_features

        self.features = numeric_features + [category_features]

    def getID(self):
        return self.full_set["id"]

    def getX(self):
        return pd.get_dummies(self.full_set[self.features])

    def getY(self):
        return self.full_set[self.y_col]

    # TODO: fix poly for full_set
    def generatePolynomialFeatures(self, poly_degree = 2, interaction = False, log = True):

        if interaction:


            self.poly = PolynomialFeatures(degree = poly_degree, include_bias = False)

            poly_fit = self.poly.fit_transform(self.full_set[self.numeric_features])

            new_features = self.poly.get_feature_names(self.numeric_features)

            new_features = list(map(lambda x: x.replace(" ", "_"), new_features))

            self.numeric_features = new_features

            self.updateFeaturesList()

            self.full_set = pd.concat([self.full_set["id"],
                self.full_set[self.y_col], 
                self.full_set[self.category_features],
                pd.DataFrame(poly_fit, columns = self.numeric_features)], axis = 1)

        else:

            new_features = []

            for power in range(2, poly_degree + 1):
                for col in self.numeric_features:

                    feature_name = col + "_" + str(power)
                    self.full_set[feature_name] = np.power(self.full_set[col], power)
                    new_features.append(feature_name)

            if log:

                for col in self.numeric_features:

                    feature_name = "log_"+col
                    self.full_set[feature_name] = np.power(self.full_set[col], power)
                    new_features.append(feature_name)

            self.numeric_features += new_features
            self.features += new_features



class TestSet(DataSet):

    def __init__(self, data, competition_type, era_cat, numeric_features, cluster_model, clusters):

        super(TestSet, self).__init__(data, competition_type)

        self.numeric_features = numeric_features

        self.full_set["cluster"] = pd.Categorical(cluster_model.assignClusters(self.full_set[self.numeric_features]), categories = clusters)

        self.features += ["cluster"]

        self.eras = era_cat
        # Another gross hack with self.category_features[0]
        self.full_set[self.category_features] = pd.Categorical(self.full_set[self.category_features], ordered = True, categories = era_cat)


class TrainSet(DataSet):

    split_index = {'train' : [], 'test' : []}

    def __init__(self, data, competition_type):

        super(TrainSet, self).__init__(data, competition_type)

        # self.reduceFeatureSpace(0.05)

        print("Estimating Clusters")
        self.cluster_model = ClusterFeature(self.full_set[self.numeric_features], None)

        cluster_id = self.cluster_model.assignClusters(self.full_set[self.numeric_features])

        self.clusters = np.unique(cluster_id)

        self.full_set["cluster"] = pd.Categorical(cluster_id, categories = self.clusters)

        self.features += ["cluster"]

        # A HACK THAT I NEED TO FIX (subset category features to get 0)
        self.eras = self.full_set[self.category_features].unique()
        self.full_set[self.category_features] = pd.Categorical(self.full_set[self.category_features],
            ordered = True, categories = self.eras)

        self.split_index = {'train' : [i for i in range(0,self.N)], 'test' : []}


    def getEras(self):

        return self.eras

    def updateSplit(self, train_ind, test_ind):

        self.split_index = {'train' : train_ind, 'test' : test_ind}

    def getY(self, train = None):

        if train is None:

            return self.full_set[self.y_col]

        elif train:
            return self.full_set[self.y_col].iloc[self.split_index["train"]]
        else:
            return self.full_set[self.y_col].iloc[self.split_index["test"]]

    def getX(self, train = None):

        if train is None:
            return pd.get_dummies(self.full_set[self.features])

        elif train:
            return pd.get_dummies(self.full_set[self.features].iloc[self.split_index["train"]])
        else:
            return pd.get_dummies(self.full_set[self.features].iloc[self.split_index["test"]])


class FeatureGenerator():

    def __init__(self, degree = 2, cluster = False):
        self.poly = PolynomialFeatures(degree, include_bias = False)




# class NumeraiCompetitionSet():

#     def __init__(self, data, competition_type):
#         self.train = TrainSet(data, competition_type)
#         self.test = TestSet(data, competition_type, train.getEras(), train.numeric_features)

#     def setPolynomial(self, degree = 2):
#         self.poly = PolynomialFeatures(degree, include_bias = False)




    # def generatePolynomialFeatures(self, poly_degree = 2):
    #     self.poly = PolynomialFeatures(degree = poly_degree, include_bias = False)

    #     output = self.poly.fit_transform(self.x_full)

    #     self.x_full = pd.DataFrame(output, columns = self.poly.get_feature_names(self.x_full.columns))

    #     self.updateSplit(self.split_index['train'], self.split_index['test'])

    # def setPolynomialFeatures(self):

    #     output = self.poly.fit_transform(self.full_set[self.features])

    #     return(pd.DataFrame(output, columns = poly.get_feature_names(self.x_full.columns)))

if __name__ == "__main__":

    df = pd.DataFrame({"era": ["a", "a", "b", "b"], "feature_one": [1,2,3,4], "feature_two": [5,6,7,8]})
    ds = DataSet(df, "test")

    print(ds.full_set)
    print(ds.features)

    print(ds.numeric_features)
    print(ds.category_features)
