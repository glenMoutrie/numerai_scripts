import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class DataSet():

    full_set = np.ndarray(None)

    poly = None

    def __init__(self, data, competition_type):

        self.full_set = data

        self.numeric_features = [f for f in list(self.full_set) if "feature" in f]

        # Test just the first four features
        # self.numeric_features = self.numeric_features[0:3]
        
        self.category_features = "era"

        self.competition_type = competition_type
        self.y_col = 'target_' + competition_type

        self.updateFeaturesList()

        self.generatePolynomialFeatures(2)

        self.N = data.shape[0]


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
    def generatePolynomialFeatures(self, poly_degree = 2):

        self.poly = PolynomialFeatures(degree = poly_degree, include_bias = False)

        poly_fit = self.poly.fit_transform(self.full_set[self.numeric_features])

        self.numeric_features = self.poly.get_feature_names(self.numeric_features)

        self.updateFeaturesList()

        self.full_set = pd.concat([self.full_set["id"],
            self.full_set[self.y_col], 
            self.full_set[self.category_features],
            pd.DataFrame(poly_fit, columns = self.numeric_features)], axis = 1)


class TestSet(DataSet):

    def __init__(self, data, competition_type, era_cat):

        super(TestSet, self).__init__(data, competition_type)

        self.eras = era_cat
        # Another gross hack with self.category_features[0]
        self.full_set[self.category_features] = pd.Categorical(self.full_set[self.category_features], categories = era_cat)


class TrainSet(DataSet):

    split_index = {'train' : [], 'test' : []}

    def __init__(self, data, competition_type):

        super(TrainSet, self).__init__(data, competition_type)

        # A HACK THAT I NEED TO FIX (subset category features to get 0)
        self.eras = self.full_set[self.category_features].unique()
        self.full_set[self.category_features] = pd.Categorical(self.full_set[self.category_features], categories = self.eras)

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




class NumeraiCompetitionSet():

    def __init__(self, data, competition_type):
        self.test = TestSet(data, competition_type)
        self.train = TrainSet(data, competition_type, test.getEras())

    def setPolynomial(self, degree = 2):
        self.poly = PolynomialFeatures(degree, include_bias = False)




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
