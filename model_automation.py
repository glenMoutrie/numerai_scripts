import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, linear_model, svm, metrics, ensemble, naive_bayes
from sklearn.model_selection import ShuffleSplit

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
