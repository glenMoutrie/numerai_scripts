import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, linear_model, svm, metrics, ensemble, naive_bayes
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import time
from datetime import datetime

# TODO: INCORPORATE ERAS INTO VALIDATION
class ModelTester():

    def __init__(self, models, eras, splits = 5, test_size = 0.25) :

        # self.appendFeatureSelection()

        self.models = models
        self.eras = eras.tolist() + ['all']

        self.splits = splits
        self.splits_performed = 0

        self.ss = ShuffleSplit(n_splits = splits, test_size = test_size)

        index = [(i, j, k) for i in range(1, splits + 1) for j in self.eras for k in models.keys()]
        index = pd.MultiIndex.from_tuples(index, names = ['split', 'era', 'model'])

        # index = [(i, j) for i in range(1, splits + 1) for j in models.keys()]
        # index = pd.MultiIndex.from_tuples(index, names = ['split', 'model'])

        self.measures =['duration', 'log_loss', 'precision', 'recall', 'f1', 'auc']

        self.model_performance = pd.DataFrame(columns = self.measures, index = index)

        print(self.model_performance)

        self.best_model = None

    def appendFeatureSelection(self):
        for name, model in zip(self.models.keys(), self.models.values()):
            self.models[name] = Pipeline([('feature selection', SelectFromModel(model))
                ('model', model)])

    def testAllSplits(self, data):

        for train_i, test_i in self.ss.split(data.getX()):

            print("\n\nTesting split: " + str(self.splits_performed))

            data.updateSplit(train_i, test_i)
            self.splits_performed += 1

            self.testAllModels(data,  self.models)



    def testAllModels(self, data, models):

        mp_update = {model: self.testModel(data, models.get(model), model) for model in models.keys()}


        for update in mp_update.keys():
            for era in self.eras:
                self.model_performance.loc[(self.splits_performed, era, update)] = mp_update[update][era]

    def testModel(self, data, model, name, verbose = True):

        t1 = time.time()

        if verbose:
            print("Testing " + name + ":")

        if name in ['xgboost']:

            try:
                model.fit(data.getX(True), data.getY(True))

            except:
                print(name + " failed")
                return(np.nan)

        else:

            model.fit(data.getX(True), data.getY(True).round())

        y_prediction = model.predict_proba(data.getX(False))

        results = y_prediction[:, 1]

        metrics = {}

        for era in self.eras:


            if era == 'all':
                all_metrics = self.getMetrics(data.getY(False).to_numpy(),  results, t1)
                metrics[era] = all_metrics

            else:

                ind = np.isin(data.split_index['test'], np.argwhere(data.getEraIndex(era).to_numpy()))

                metrics[era] = self.getMetrics(data.getY(False, era).to_numpy(), results[ind], t1)
                

        if verbose:
            print("Time taken: " + str(all_metrics['duration']) + 
                "\nLog loss: " + str(all_metrics['log_loss']) + 
                "\nPrecision: " + str(all_metrics['precision']) +
                "\nRecall: " + str(all_metrics['recall']))

        return metrics

    def getMetrics(self,  observed, results, t1):


        stop_metrics = observed.size <= 1

        stop_metrics = stop_metrics or (np.unique(results).size <= 1)

        stop_metrics = stop_metrics or (np.unique(observed).size <= 1)

        if stop_metrics:
            return(dict(zip(self.measures, [np.nan] * len(self.measures))))

        duration = time.time() - t1

        try:

            log_loss = metrics.log_loss(observed.round(), results)

            precision = metrics.precision_score(observed.round(), results.round())

            recall = metrics.recall_score(observed.round(), results.round())

            f1 = metrics.f1_score(observed.round(), results.round())

            auc = metrics.roc_auc_score(observed.round(), results.round())

        except UndefinedMetricWarning as undefined_metric:
            pass

        output = dict(zip(self.measures, [duration, log_loss, precision, recall, f1, auc]))


        return output



    def getBestModel(self):

        # self.best_model = 'xgboost'

        # return self.models['xgboost']

        print(self.model_performance)

        self.model_performance = self.model_performance.reset_index()

        self.best_model = self.model_performance\
        .groupby('model')\
        .apply(lambda x: x[self.measures[1:]].agg('mean'))\
        .apply(lambda x: x.agg('rank'))\
        .apply(lambda x: x.sum(), axis = 1)\
        .idxmin()

        print("\n\nBest model: " + self.best_model)

        self.logMetrics()

        return self.best_model

    def getBestPrediction(self, train_data, test_data):

        model = self.models[self.getBestModel()]

        if self.best_model in ['xgboost']:

            model.fit(train_data.getX(), train_data.getY())

        else:

            model.fit(train_data.getX(), train_data.getY().round())

        output =  model.predict_proba(test_data.getX())[:,1]

        # print("Test Log Loss: " + str(metrics.log_loss(test_data.getY("test"), model.predict_proba(test_data.getX("test"))[:,1])))

        return output

    def logMetrics(self):

        self.model_performance.to_csv("logs/model_performance/metric_log_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")
