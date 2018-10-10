import pandas as pd
import numpy as np

class DataSet():

    full_set = np.ndarray(None)

    poly = None

    def __init__(self, data, competition_type):

        self.full_set = data

        self.features = [f for f in list(self.full_set) if "feature" in f]
        self.competition_type = competition_type
        self.y_col = 'target_' + competition_type

        self.N = data.shape[0]

    def getID(self):
        return self.full_set["id"]

    def getX(self):
        return self.full_set[self.features]

    def getY(self):
        return self.full_set[self.y_col]

    # TODO: fix poly for full_set
    # def generatePolynomialFeatures(self, poly_degree = 2):
    #     self.poly = PolynomialFeatures(degree = poly_degree, include_bias = False)

    #     output = self.poly.fit_transform(self.x_full)

    #     self.x_full = pd.DataFrame(output, columns = self.poly.get_feature_names(self.x_full.columns))

    #     self.updateSplit(self.split_index['train'], self.split_index['test'])

    # def setPolynomialFeatures(self):

    #     output = self.poly.fit_transform(self.full_set[self.features])

    #     return(pd.DataFrame(output, columns = poly.get_feature_names(self.x_full.columns)))

class TestSet(DataSet):

    def __init__(self, data, competition_type):

        super(TestSet, self).__init__(data, competition_type)

class TrainSet(DataSet):

    split_index = {'train' : [], 'test' : []}

    def __init__(self, data, competition_type):

        super(TrainSet, self).__init__(data, competition_type)


        self.split_index = {'train' : [i for i in range(0,self.N)], 'test' : []}


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
            return self.full_set[self.features]

        elif train:
            return self.full_set[self.features].iloc[self.split_index["train"]]
        else:
            return self.full_set[self.features].iloc[self.split_index["test"]]
