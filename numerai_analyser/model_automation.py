import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, linear_model, svm, metrics, ensemble, naive_bayes
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import time

class ModelTester():

    def __init__(self, models, splits = 5, test_size = 0.25) :

        # self.appendFeatureSelection()

        self.models = models
        self.splits = splits
        self.splits_performed = 0

        self.ss = ShuffleSplit(n_splits = splits, test_size = test_size)
        self.model_performance = pd.DataFrame(columns = list(self.models.keys()))
        self.best_model = None

    def appendFeatureSelection(self):
        for name, model in zip(self.models.keys(), self.models.values()):
            print(model)
            self.models[name] = Pipeline([('feature selection', SelectFromModel(model))
                ('model', model)])

    def testAllSplits(self, data):

        for train_i, test_i in self.ss.split(data.getX()):

            data.updateSplit(train_i, test_i)
            self.splits_performed += 1

            self.testAllModels(data,  self.models)

            print("Splits tested: " + str(self.splits_performed))



    def testAllModels(self, data, models):

        mp_update = {model: self.testModel(data, models.get(model), model) for model in models.keys()}

        self.model_performance = self.model_performance.append(mp_update, ignore_index = True)


    def testModel(self, data, model, name, verbose = True):

        t1 = time.time()

        if verbose:
            print("Testing " + name , end = "")

        # print(data.getX(True))

        if name in ['gradientBoosting', 'xgboost']:

            model.fit(data.getX(True), data.getY(True))

        else:

            model.fit(data.getX(True), data.getY(True).round())

        y_prediction = model.predict_proba(data.getX(False))

        results = y_prediction[:, 1]

        duration = time.time() - t1

        log_loss = metrics.log_loss(data.getY(False).round(), results)

        if verbose:
            print("Time taken: " + str(duration) + " Log loss: " + str(log_loss))

        return log_loss

    def getBestModel(self):

        print(self.model_performance.apply(np.mean))
        print(self.model_performance.apply(np.mean).idxmin())
        self.best_model = self.models[self.model_performance.apply(np.mean).idxmin()]

        print("Best model: " + self.model_performance.apply(np.mean).idxmin() + "; Logistic Loss = "  + str(self.model_performance.apply(np.mean).min()))

        return self.best_model

    def getBestPrediction(self, train_data, test_data):

        model = self.getBestModel()

        model.fit(train_data.getX(), train_data.getY().round())

        output =  model.predict_proba(test_data.getX())[:,1]

        # print("Test Log Loss: " + str(metrics.log_loss(test_data.getY("test"), model.predict_proba(test_data.getX("test"))[:,1])))

        return output
