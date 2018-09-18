#!numerai/bin/python


import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, linear_model, svm, metrics, ensemble, naive_bayes
from sklearn.model_selection import ShuffleSplit


# Plan:
# 1) create a data loader that creates a NumeraiData class
# 2) Define a NumeraiData Class with features, x data and y data
# 3) finish the model tester that returns the model results, estimating in parallel using multiprocessing
# 4) automate with numerapi and other such tools

# CURRENT TODO: FIX BUGS IN UPDATE

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




class DataLoader():

    training_data_file = 'numerai_training_data.csv'
    test_data_file = 'numerai_tournament_data.csv'



    def __init__(self, loc, date) :
        self.loc = loc + "numerai_datasets_" + date + "/"
        self.date = date

    def read(self):
        self.train = pd.read_csv(self.loc + self.training_data_file, header = 0)
        self.test = pd.read_csv(self.loc + self.test_data_file, header = 0)

        self.features = [f for f in list(self.train) if "feature" in f]

    def write(self, output):
        output.to_csv(self.loc + "predictions.csv", index = False)

    def getData(self, competition_type):
        self.train = DataSet(self.train[self.features], self.train['target_' + competition_type])
        self.test = self.test

        return self.train, self.test




class ModelTester():

    models = {'logistic' : linear_model.LogisticRegression(n_jobs=1),
                  'naiveBayes' : naive_bayes.GaussianNB()}
                  # 'randomForest' : ensemble.RandomForestClassifier(),
                  # 'extraTrees' : ensemble.ExtraTreesClassifier(),
                  # 'gradientBoosting' : ensemble.GradientBoostingClassifier(),
                  # 'adaBoost' : ensemble.AdaBoostClassifier()}

    def __init__(self, splits = 5, test_size = 0.25) :
        self.splits = splits
        self.splits_performed = 0
        self.ss = ShuffleSplit(n_splits = splits, test_size = test_size)
        self.model_performance = pd.DataFrame(columns = list(self.models.keys()))
        self.best_model = None


    def testAllSplits(self, data):

        for train_i, test_i in self.ss.split(data.getXFull()):

            data.updateSplit(train_i, test_i)
            self.splits_performed += 1

            self.testAllModels(data,  self.models)

            print("Splits tested: " + str(self.splits_performed))
            print("Test acc: " + str(self.model_performance) + "\n")



    def testAllModels(self, data, models):

        mp_update = {model: self.testModel(data, models.get(model)) for model in models.keys()}

        self.model_performance = self.model_performance.append(mp_update, ignore_index = True)

    def testModel(self, data, model):

        model.fit(data.getX(True), data.getY(True))

        y_prediction = model.predict_proba(data.getX(False))

        results = y_prediction[:, 1]

        return metrics.log_loss(data.getY(False), results)

    def getBestModel(self):

        print(self.model_performance)
        self.best_model = self.models[self.model_performance.apply(np.mean).idxmin()]

        print("Best model: " + self.model_performance.apply(np.mean).idxmin() + "; Logistic Loss = "  + str(self.model_performance.apply(np.mean).min()))

        return self.best_model

    def getBestPrediction(self, train_data, test_data):

        model = self.getBestModel()

        model.fit(train_data.getXFull(), train_data.getYFull())

        return model.predict_proba(test_data)[:,1]







if __name__ == '__main__':

    dl = DataLoader("datasets/", "17_07_2018")
    dl.read()
    
    train, test = dl.getData('bernie')
    
    tester = ModelTester(5, 0.25)

    tester.testAllSplits(train)

    results = tester.getBestPrediction(train, test[dl.features])

    results_df = pd.DataFrame(data={'probability': results})
    results_df = pd.DataFrame(test["id"]).join(results_df)
    dl.write(results_df)




    # predictData('bernie')
