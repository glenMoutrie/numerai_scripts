import pandas as pd
import numpy as np

from joblib import Parallel, delayed

from sklearn.model_selection import ShuffleSplit

import time

from .model_metrics import ModelMetrics
from .neutralize_normalize import auto_neutralize_normalize

from .trained_model_io import NumeraiTrainedOutput
from .model_factory import ModelFactory

import traceback
import warnings
import sys


# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#
#     log = file if hasattr(file,'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))
#
# warnings.showwarning = warn_with_traceback


class ModelTester():

    def __init__(self, model_factory, eras, config, splits=5, test_size=0.25):

        self.config = config

        self.model_factory = model_factory

        self.eras = eras.tolist() + ['all']

        self.splits = splits
        self.splits_performed = 0

        self.all_ranks = []
        self.all_valid_checks = []

        self.ss = ShuffleSplit(n_splits=splits, test_size=test_size)

        index = [(i, j, k) for i in range(1, splits + 1) for j in self.eras for k in
                 (list(self.model_factory.models.keys()) + ['ensemble'])]
        index = pd.MultiIndex.from_tuples(index, names=['split', 'era', 'model'])

        self.model_performance = pd.DataFrame(columns=ModelMetrics.measures, index=index, dtype=np.float)

        self.best_model = None

        self.parallel = Parallel(n_jobs=config.n_cores, require = 'sharedmem')

    '''
    The functions below are all for testing and training on the specified number of splits
    '''

    def testAllSplits(self, data):

        for train_i, test_i in self.ss.split(data.getX()):
            self.config.logger.info("TESTING SPLIT: " + str(self.splits_performed))

            data.updateSplit(train_i, test_i)
            self.splits_performed += 1

            self.testAllModels(data, self.model_factory.models)

        self.all_ranks = pd.concat(self.all_ranks)
        self.all_valid_checks = pd.concat(self.all_valid_checks)

    def testAllModels(self, data, models):

        self.predictions = pd.DataFrame(columns=['era'] + list(models.keys()) + ['ensemble'],
                                        index=data.full_set[data.full_set.data_type == 'test'].index)

        self.predictions['era'] = data.full_set[data.full_set.data_type == 'test'].era.to_numpy()

        mp_update = self.parallel(delayed(self.testModel)(data, model)
                                  for model in models.keys() if not model.startswith('DNN'))

        # Keras models can't be compiled in parallel
        mp_update = mp_update + [self.testModel(data, model)
                                 for model in models.keys() if model.startswith('DNN')]

        mp_update = {i[1]: i[0] for i in mp_update}

        for update in mp_update.keys():
            for era in self.eras:
                self.model_performance.loc[(self.splits_performed, era, update)] = mp_update[update][era]

        self.config.logger.info('Estimating ensemble predictions')

        self.calculateEnsemblePredictions()

        self.predictions.astype(dict(zip(list(models.keys()) + ['ensemble'], ['float64'])))

        for era in self.eras:

            if era == 'all':

                _metrics = ModelMetrics.getMetrics(observed=data.getY(data_type='test').to_numpy(),
                                                   results=self.predictions['ensemble'].to_numpy(),
                                                   t1=np.nan)

            else:



                _metrics = ModelMetrics.getMetrics(observed=data.getY(data_type='test', era=era).to_numpy(),
                                                   results=self.predictions.loc[
                                                       self.predictions.era == era, 'ensemble'].to_numpy(),
                                                   t1=np.nan)

            self.model_performance.loc[(self.splits_performed, era, 'ensemble')] = _metrics

    def testModel(self, data, name):
        '''
        Tests an individual model, collecting performance metrics by era.
        '''

        t1 = time.time()

        self.config.logger.info("TESTING " + name.upper() + "")

        results = ModelFactory.estimate_model(model = self.model_factory[name],
                                                    model_name=name,
                                                    train_data=data,
                                                    test_data=None,
                                                    data_type_train='train',
                                                    data_type_test='test')

        self.predictions[name] = results

        metrics = {}

        for era in self.eras:

            if era == 'all':
                all_metrics = ModelMetrics.getMetrics(data.getY(data_type='test').to_numpy(), results, t1)
                metrics[era] = all_metrics

            else:

                ind = data.full_set[data.full_set.data_type == 'test'].era == era

                check = data.getY(data_type='test', era=era).to_numpy().size <= 1
                check = check and results[ind].size <= 1

                if check:
                    self.config.logger.info("No data for era: {0}, model: {1}, split: {2}".format(era, name, self.splits_performed))
                    metrics[era] = ModelMetrics.defaultOutput()

                else:
                    metrics[era] = ModelMetrics.getMetrics(data.getY(data_type='test', era=era).to_numpy(), results[ind], t1)

        log = name.upper() + " METRICS:\t"
        log += "Time taken: {:.2f}".format(all_metrics['duration']) + ", "
        log += "Log loss: {:.2f}".format(all_metrics['log_loss']) + ", "
        log += "Precision: {:.2f}".format(all_metrics['precision']) + ", "
        log += "Recall: {:.2f}".format(all_metrics['recall'])

        self.config.logger.info(log)

        return metrics, name

    def getBestModel(self):

        # The below code is sometimes used for forcing a model...
        # self.best_model = 'xgboost'
        # return self.best_model

        self.model_performance = self.model_performance.reset_index()

        # rank = self.getModelRank(self.model_performance)

        rank = self.getModelSharpe(self.model_performance)

        self.config.logger.info('Model ranking: ')

        self.config.logger.info(str(rank.sort_values()))

        self.config.logger.info('Ignoring models that are not cross validated: ' + (', '.join(self.model_factory.saved_model)))

        self.best_model = rank.loc[~rank.index.isin(self.model_factory.saved_model)].idxmax()


        self.config.logger.info("Best model: " + self.best_model)

        self.logMetrics()

        return self.best_model

    def getModelRank(self, model_performance):

        return model_performance \
            .assign(log_loss=lambda x: x.log_loss * -1) \
            .groupby('model') \
            .apply(lambda x: x[ModelMetrics.target_measures].agg('mean')) \
            .apply(lambda x: x.rank(pct=True, method='first')) \
            .apply(lambda x: x.mean(), axis=1)

    def getModelSharpe(self, model_performance):

        cov_mean = model_performance \
            .groupby(['model']) \
            .apply(lambda x: x.num_cov.mean())

        cov_sd = model_performance \
            .groupby(['model']) \
            .apply(lambda x: x.num_cov.std())

        return cov_mean / cov_sd

    @classmethod
    def getPrediction(cls, config, train_data, test_data, name, weights, trained_models):

        if name == 'ensemble':

            output = cls.ensemblePrediction(config, train_data, test_data, weights, trained_models)

        else:

            output = ModelFactory.estimate_model(model = trained_models[name], model_name=name,
                                                       train_data=train_data, test_data=test_data)


        metrics_orig = ModelMetrics.getNumeraiScoreByEra(test_data.getY(data_type = 'validation'),
                                                               output[test_data.full_set.data_type == 'validation'],
                                                               test_data.getEras())

        output = auto_neutralize_normalize(output, test_data, config.n_cores, config.logger)

        config.logger.info("Predictions summary statistics:")
        config.logger.info(str(output.describe()))

        metrics_normalized = ModelMetrics.getNumeraiScoreByEra(test_data.getY(data_type = 'validation'),
                                                               output[test_data.full_set.data_type == 'validation'],
                                                               test_data.getEras())

        config.logger.info("Original Numerai Score:")
        config.logger.info("Validation Correlation: {0}\nValidation Sharpe: {1}" \
                                .format(metrics_orig['correlation'], metrics_orig['sharpe']))

        config.logger.info("Neutralized Numerai Score:")
        config.logger.info("Validation Correlation: {0}\nValidation Sharpe: {1}" \
                                .format(metrics_normalized['correlation'], metrics_normalized['sharpe']))

        return output


    def train_all_models(self, train):

        self.trained_models = {}

        for i in (self.model_factory.models.keys()):
            self.config.logger.info('TRAINING ' + i.upper() + ' ON FULL DATA SET')

            self.trained_models[i] = ModelFactory.estimate_model(model = self.model_factory[i], model_name=i,
                                                                       train_data=train, return_model = True)


    def getTrainedOutput(self, features):

        return NumeraiTrainedOutput(features, self.trained_models, self.getEnsembleWeights(), self.config.run_param, self.config.test_param)

    def logMetrics(self):

        self.model_performance.to_csv(self.config.metric_loc_file)

    '''

    ENSEMBLE MODEL CALCULATION

    All of the functions below are used to calculate the ensemble model.

    As it would be costly to estimate all of the models each time this uses the predictions that have been already made on 
    that test/split.

    '''

    @staticmethod
    def weightedAveragePreds(weights, predictions):

        weights = weights.fillna(0)

        output = predictions[weights.index.to_list()] \
            .apply(lambda x: np.average(x, weights=weights), axis=1)

        return output

    def calculateEnsemblePredictions(self):

        index = self.model_performance.index.get_level_values('split') == self.splits_performed
        index = index & (self.model_performance.index.get_level_values('model') != 'ensemble')

        rank = self.model_performance \
            .loc[index] \
            .groupby('era') \
            .apply(self.getModelRank)

        viable = self.model_performance \
            .loc[index] \
            .groupby(['era', 'model']) \
            .apply(lambda x: x.num_cov.mean() > 0) \
            .rename('viable') \
            .reset_index()

        self.all_ranks.append(rank)
        self.all_valid_checks.append(viable)

        for i in rank.index.unique():

            if viable[viable.era == i]['viable'].any():

                viable_ranks = rank.loc[i][viable.loc[(viable.era == i) & (viable['viable'])].model]
                self.predictions.loc[self.predictions.era == i, 'ensemble'] = self.weightedAveragePreds(viable_ranks,
                                                                                                        self.predictions.loc[
                                                                                                            self.predictions.era == i])

            else:

                _best_model = rank.loc[i].idxmax(axis=1)

                if str(_best_model) == 'nan':
                    _best_model = rank.loc['all'].idxmax(axis=1)

                self.predictions.loc[self.predictions.era == i, 'ensemble'] = self.predictions.loc[
                    self.predictions.era == i, _best_model]

        rank['split'] = self.splits_performed
        viable['split'] = self.splits_performed

    def getEnsembleWeights(self):
        erax_validity = self.all_valid_checks \
            .astype({'viable': 'int32'}) \
            .groupby('model')['viable'] \
            .agg('mean') \
            .apply(lambda x: x >= 0.2)

        models = erax_validity[erax_validity].index.tolist()

        if not erax_validity.any():

            self.config.logger.warning("No valid models, using all")

            models = erax_validity.index.tolist()

        else:

            self.config.logger.info("Valid models: " + ", ".join(erax_validity[erax_validity].index.tolist()))

        weights = self.all_ranks[list(models)].agg(np.nanmean)

        if weights.idxmax() == 'xgbreg_costly':
            weights['xgbreg_costly'] = weights.drop('xgbreg_costly').max()

        return weights

    @classmethod
    def ensemblePrediction(cls, config, train, test, weights = None, trained_models = None):

        models = weights.index.to_list()

        predictions = pd.DataFrame(columns=models, index=[i for i in range(test.N)])

        for i in models:
            config.logger.info('PROVIDING PREDICTIONS FOR ' + i.upper())

            predictions[i] = ModelFactory.estimate_model(model = trained_models[i], model_name = i, train_data=train,
                                                         test_data=test, predict_only= True)

        output = cls.weightedAveragePreds(weights, predictions)

        return output
