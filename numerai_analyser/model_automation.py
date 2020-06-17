import pandas as pd
import numpy as np
import scipy as sp

from sklearn import preprocessing, linear_model, svm, metrics, ensemble, naive_bayes
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import time
from datetime import datetime

from xgboost import XGBClassifier, XGBRegressor
from .DNN import DNNVanilla
from .model_metrics import ModelMetrics


def _neutralize(df, columns, by, proportion=1.0):

    scores = df[columns]
    exposures = df[by].values
    scores = scores - proportion * exposures.dot(np.linalg.pinv(exposures).dot(scores))

    scores[scores < 0] = 0
    scores[scores > 1] = 1

    return scores / scores.std()
    
def _normalize(df):
    X = (df.rank(method="first") - 0.5) / len(df)
    return sp.stats.norm.ppf(X)

def _normalize_and_neutralize(df, columns, by, proportion=1.0):
    # Convert the scores to a normal distribution
    df[columns] = _normalize(df[columns])
    df[columns] = _neutralize(df, columns, by, proportion)
    return df[columns]

def normalizeAndNeutralize(predictions, test, proportion = 1.0):

    X = test.getX(data_type = None, all_features = True)

    output = X\
    .assign(PREDS = predictions,
        ERA = test.full_set.era)\
    .groupby('ERA')\
    .apply(lambda x: _normalize_and_neutralize(x, ["PREDS"], test.all_features))['PREDS']

    return output

class ModelTester():

    def __init__(self, models, eras, config, splits = 5, test_size = 0.25) :

        self.config = config

        self.models = models
        self.eras = eras.tolist() + ['all']

        self.splits = splits
        self.splits_performed = 0

        self.all_ranks = []
        self.all_valid_checks = []

        self.ss = ShuffleSplit(n_splits = splits, test_size = test_size)

        index = [(i, j, k) for i in range(1, splits + 1) for j in self.eras for k in (list(models.keys()) + ['ensemble'])]
        index = pd.MultiIndex.from_tuples(index, names = ['split', 'era', 'model'])

        self.model_metrics = ModelMetrics()

        self.model_performance = pd.DataFrame(columns = self.model_metrics.measures, index = index)

        self.best_model = None


    '''
    The functions below are all for testing and training on the specified number of splits
    '''

    def testAllSplits(self, data):

        for train_i, test_i in self.ss.split(data.getX()):

            self.config.logger.info("TESTING SPLIT: " + str(self.splits_performed))

            data.updateSplit(train_i, test_i)
            self.splits_performed += 1

            self.testAllModels(data,  self.models)

        self.all_ranks = pd.concat(self.all_ranks)
        self.all_valid_checks = pd.concat(self.all_valid_checks)

        print(self.all_ranks)
        print(self.all_valid_checks)

    def testAllModels(self, data, models):

        self.predictions = pd.DataFrame(columns = ['era'] + list(models.keys()) + ['ensemble'],
            index = data.split_index['test'])

        self.predictions['era'] = data.full_set.iloc[data.split_index['test']].era.to_numpy()

        mp_update = {model: self.testModel(data, models.get(model), model) for model in models.keys()}

        for update in mp_update.keys():
            for era in self.eras:
                self.model_performance.loc[(self.splits_performed, era, update)] = mp_update[update][era]

        self.config.logger.info('Estimating ensemble predictions')

        self.calculateEnsemblePredictions()

        self.predictions.astype(dict(zip(list(models.keys()) + ['ensemble'], ['float64'])))

        

        for era in self.eras:

            if era == 'all':
                
                _metrics == self.model_metrics.getMetrics(observed = data.getY(False).to_numpy(),
                    results = self.predictions['ensemble'].to_numpy(),
                    t1 = np.nan)

            else:

                _metrics = self.model_metrics.getMetrics(observed = data.getY(False, era).to_numpy(), 
                results = self.predictions.loc[self.predictions.era == era, 'ensemble'].to_numpy(), 
                t1 = np.nan)

            self.model_performance.loc[(self.splits_performed, era, 'ensemble')] = _metrics

    

    '''
    Tests an individual model, collecting performance metrics by era.
    '''
    def testModel(self, data, model, name):

        t1 = time.time()

        self.config.logger.info("TESTING " + name.upper() + "")

        if name in ['xgboostReg', 'DNN']:

            model.fit(data.getX(True), data.getY(True))
            results = model.predict(data.getX(False))

        elif name in ['xgboost_num', 'DNN_full']:

            model.fit(data.getX(train = True, all_features = True), data.getY(True))
            results = model.predict(data.getX(train = False, all_features = True))

        else:

            model.fit(data.getX(True), data.getY(True).round())
            y_prediction = model.predict_proba(data.getX(False))
            results = y_prediction[:, 1]

        self.predictions[name] = results

        metrics = {}

        for era in self.eras:


            if era == 'all':
                all_metrics = self.model_metrics.getMetrics(data.getY(False).to_numpy(),  results, t1)
                metrics[era] = all_metrics

            else:

                ind = np.isin(data.split_index['test'], np.argwhere(data.getEraIndex(era).to_numpy()))

                metrics[era] = self.model_metrics.getMetrics(data.getY(False, era).to_numpy(), results[ind], t1)
                
        log = name.upper() + " METRICS:\t"
        log += "Time taken: {:.2f}".format(all_metrics['duration']) + ", "
        log += "Log loss: {:.2f}".format(all_metrics['log_loss']) + ", "
        log += "Precision: {:.2f}".format(all_metrics['precision']) + ", "
        log += "Recall: {:.2f}".format(all_metrics['recall'])

        self.config.logger.info(log)

        return metrics


    def getBestModel(self):

        # The below code is sometimes used for forcing a model...
        # self.best_model = 'xgboost'
        # return self.models['xgboost']

        print(self.model_performance)

        self.model_performance = self.model_performance.reset_index()

        # rank = self.getModelRank(self.model_performance)

        rank = self.getModelSharpe(self.model_performance)

        self.config.logger.info('Model ranking: ')

        print(rank.sort_values())

        self.best_model = rank.idxmax()

        self.config.logger.info("Best model: " + self.best_model)

        self.logMetrics()

        return self.best_model


    def getModelRank(self, model_performance):

        return model_performance\
        .assign(log_loss = lambda x: x.log_loss * -1)\
        .groupby('model')\
        .apply(lambda x: x[self.model_metrics.target_measures].agg('mean'))\
        .apply(lambda x: x.rank(pct = True, method = 'first'))\
        .apply(lambda x: x.mean(), axis = 1)

    def getModelSharpe(self, model_performance):

        cov_mean = model_performance\
        .groupby(['model'])\
        .apply(lambda x: x.num_cov.mean())

        cov_sd = model_performance\
        .groupby(['model'])\
        .apply(lambda x: x.num_cov.std())

        return cov_mean/cov_sd


    def getBestPrediction(self, train_data, test_data):

        name = self.getBestModel()

        if name in ['xgboostReg', 'DNN']:

            model = self.models[name]

            model.fit(train_data.getX(), train_data.getY())
            output = model.predict(test_data.getX())

        elif name == 'ensemble':

            output = self.ensemblePrediction(train_data, test_data)

        elif name in ['xgboost_num', 'DNN_full']:

            model = self.models[name]

            model.fit(train_data.getX(train = None, all_features = True), train_data.getY())
            output = model.predict(test_data.getX(data_type = None, all_features = True))

        else:

            model = self.models[name]

            model.fit(train_data.getX(), train_data.getY().round())
            y_prediction = model.predict_proba(test_data.getX())
            output = y_prediction[:, 1]

        # print(output.describe())

        # output = normalizeAndNeutralize(output, test_data, 0.5)

        # print(output.describe())

        return output


    def logMetrics(self):

        self.model_performance.to_csv(self.config.metric_loc_file)


    '''

    ENSEMBLE MODEL CALCULATION

    All of the functions below are used to calculate the ensemble model.

    As it would be costly to estimate all of the models each time this uses the predictions that have been already made on 
    that test/split.
    
    '''

    def weightedAveragePreds(self, weights, predictions):

        weights = weights.fillna(0)

        output = predictions[weights.index.to_list()]\
        .apply(lambda x: np.average(x, weights = weights), axis = 1)

        return output


    def calculateEnsemblePredictions(self):

        index = self.model_performance.index.get_level_values('split') == self.splits_performed
        index = index & (self.model_performance.index.get_level_values('model') != 'ensemble')

        rank = self.model_performance\
        .loc[index]\
        .groupby('era')\
        .apply(self.getModelRank)

        viable = self.model_performance\
        .loc[index]\
        .groupby(['era', 'model'])\
        .apply(lambda x: x.auc.min() >= 0.5 and x.rsq.min() > 0 and x.f1.min() > 0.5)\
        .rename('viable')\
        .reset_index()

        self.all_ranks.append(rank)
        self.all_valid_checks.append(viable)

        for i in rank.index.unique():

            if viable[viable.era == i]['viable'].any():

                viable_ranks = rank.loc[i][viable.loc[(viable.era == i) & (viable['viable'])].model]
                self.predictions.loc[self.predictions.era == i, 'ensemble'] = self.weightedAveragePreds(viable_ranks, self.predictions.loc[self.predictions.era == i])

            else:

                _best_model = rank.loc[i].idxmax(axis = 1)

                if str(_best_model) == 'nan':

                    _best_model = rank.loc['all'].idxmax(axis = 1)

                self.predictions.loc[self.predictions.era == i, 'ensemble'] = self.predictions.loc[self.predictions.era == i, _best_model]

        rank['split'] = self.splits_performed
        viable['split'] = self.splits_performed
    

    def ensemblePrediction(self, train, test):

        test_eras = test.full_set.era.unique()

        erax_validity = self.all_valid_checks\
        .astype({'viable':'int32'})\
        .groupby('model')['viable']\
        .agg('mean')\
        .apply(lambda x: x >= 0.2)

        models = erax_validity[erax_validity].index.tolist()

        if not erax_validity.any():

            self.config.logger.warning("No valid models, using all")

            models = erax_validity.index.tolist()

        weights = self.all_ranks[list(models)].agg(np.nanmean)

        predictions = pd.DataFrame(columns = models, index = [i for i in range(test.N)])


        for i in models:

            self.config.logger.info('TRAINING ' + i.upper())

            model = self.models[i]

            if i in ['xgboostReg', 'DNN']:

                model.fit(train.getX(), train.getY())
                results = model.predict(test.getX())

            elif i in ['xgboost_num', 'DNN_full']:

                model = self.models[i]

                model.fit(train.getX(train = None, all_features = True), train.getY())
                results = model.predict(test.getX(data_type = None, all_features = True))

            else:

                model.fit(train.getX(), train.getY().round())
                y_prediction = model.predict_proba(test.getX())
                results = y_prediction[:, 1]

            predictions[i] = results

        output = self.weightedAveragePreds(weights, predictions)

        return(output)
