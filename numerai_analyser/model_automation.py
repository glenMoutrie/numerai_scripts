import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, linear_model, svm, metrics, ensemble, naive_bayes
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor
import time
from datetime import datetime

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

        self.measures =['duration', 'log_loss', 'corr', 'rsq', 'num_cov', 'precision', 'recall', 'f1', 'auc']

        self.model_performance = pd.DataFrame(columns = self.measures, index = index)

        print(self.model_performance)

        self.best_model = None

    # def appendFeatureSelection(self):
    #     for name, model in zip(self.models.keys(), self.models.values()):
    #         self.models[name] = Pipeline([('feature selection', SelectFromModel(model))
    #             ('model', model)])

    def testAllSplits(self, data):

        for train_i, test_i in self.ss.split(data.getX()):

            print("\n\nTesting split: " + str(self.splits_performed))

            data.updateSplit(train_i, test_i)
            self.splits_performed += 1

            self.testAllModels(data,  self.models)

    def numeraiScore(self, train, test):
        return(np.corrcoef(pd.Series(train).rank(pct = True, method = 'first'), test)[0,1])

    def testAllModels(self, data, models):

        mp_update = {model: self.testModel(data, models.get(model), model) for model in models.keys()}


        for update in mp_update.keys():
            for era in self.eras:
                self.model_performance.loc[(self.splits_performed, era, update)] = mp_update[update][era]

    def testModel(self, data, model, name, verbose = True):

        t1 = time.time()

        if verbose:
            print("Testing " + name + ":")

        if name == 'xgboostReg':

            model.fit(data.getX(True), data.getY(True))
            results = model.predict(data.getX(False))

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
        
        binary_results = getBinaryPred(results)

        stop_metrics = observed.size <= 1

        stop_metrics = stop_metrics or (np.unique(results).size <= 1)

        stop_metrics = stop_metrics or (np.unique(observed).size <= 1)

        stop_metrics = stop_metrics or (np.unique(binary_results).size <= 1)

        if stop_metrics:
            return(dict(zip(self.measures, [np.nan] * len(self.measures))))

        duration = time.time() - t1


        log_loss = metrics.log_loss(observed.round(), results)

        corr = np.correlate(observed, results)[0]

        num_cov = self.numeraiScore(observed, results)

        rsq = metrics.r2_score(observed, results)

        precision = metrics.precision_score(observed.round(), binary_results)

        recall = metrics.recall_score(observed.round(), binary_results)

        f1 = metrics.f1_score(observed.round(), binary_results)

        auc = metrics.roc_auc_score(observed.round(), binary_results)


        output = dict(zip(self.measures, [duration, log_loss, corr, num_cov, rsq, precision, recall, f1, auc]))


        return output



    def getBestModel(self):

        # self.best_model = 'xgboost'

        # return self.models['xgboost']

        print(self.model_performance)

        self.model_performance = self.model_performance.reset_index()

        rank = self.getModelRank(self.model_performance)

        print('Model ranking: ')

        print(rank.sort_values())

        self.best_model = rank.idxmax()

        print("\n\nBest model: " + self.best_model)

        self.logMetrics()

        return self.best_model

    def getModelRank(self, model_performance):

        return model_performance\
        .assign(log_loss = lambda x: x.log_loss * -1)\
        .groupby('model')\
        .apply(lambda x: x[self.measures[1:]].agg('mean'))\
        .apply(lambda x: x.rank(pct = True, method = 'first'))\
        .apply(lambda x: x.mean(), axis = 1)


    def getBestPrediction(self, train_data, test_data):

        name = self.getBestModel()

        model = self.models[name]

        if name == 'xgboostReg':

            model.fit(train_data.getX(), train_data.getY())
            output = model.predict(test_data.getX())

        else:

            model.fit(train_data.getX(), train_data.getY().round())
            y_prediction = model.predict_proba(test_data.getX())
            output = y_prediction[:, 1]


        # print("Test Log Loss: " + str(metrics.log_loss(test_data.getY("test"), model.predict_proba(test_data.getX("test"))[:,1])))

        return output

    def logMetrics(self):

        self.model_performance.to_csv("logs/model_performance/metric_log_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")

def getBinaryPred(values, level = 0.5):
    return(values > level)
