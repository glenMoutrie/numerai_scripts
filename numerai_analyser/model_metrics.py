import pandas as pd
import numpy as np
from sklearn import metrics
import time

class ModelMetrics():

    measures = ['duration', 'log_loss', 'corr', 'rsq', 'num_cov', 'precision', 'recall', 'f1', 'auc']

    target_measures = measures[1:]

    def getBinaryPred(self, values, level = 0.5):
        return(values > level)

    def defaultOutput(self):
        return(dict(zip(self.measures, [np.nan] * len(self.measures))))

    def numeraiScore(self, train, test):

        train = pd.Series(train, dtype = 'f').rank(pct = True, method = 'first').to_numpy()

        return(np.corrcoef(train, test)[0,1])

    def getNumeraiScoreByEra(self, target, predictions, era):

        measures = pd.DataFrame({'preds': target, 'preds_neutralized': predictions}) \
            .groupby(era) \
            .apply(lambda x: self.numeraiScore(x.preds, x.preds_neutralized)) \
            .agg(['mean', 'std'])

        output = {}
        output['correlation'] = measures['mean']
        output['sharpe'] = measures['mean'] / measures['std']

        return output

    def checkMetricViability(self, observed, results):

        stop_metrics = observed.size <= 1

        stop_metrics = stop_metrics or (np.unique(results).size <= 1)

        stop_metrics = stop_metrics or (np.unique(observed).size <= 1)

        stop_metrics = stop_metrics or (np.unique(observed.round()).size <= 1)

        stop_metrics = stop_metrics or (np.unique(self.getBinaryPred(results)).size <= 1)

        return(stop_metrics)


    def getMetrics(self,  observed, results, t1):

        stop_metrics = self.checkMetricViability(observed, results)

        output = self.defaultOutput()

        binary_results = self.getBinaryPred(results)

        if stop_metrics:
            return(output)

        output['duration'] = time.time() - t1

        output['log_loss'] = metrics.log_loss(observed.round(), results)

        output['corr'] = np.correlate(observed, results)[0]

        output['num_cov'] = self.numeraiScore(results, observed)

        output['rsq'] = metrics.r2_score(observed, results)

        output['precision'] = metrics.precision_score(observed.round(), binary_results)

        output['recall'] = metrics.recall_score(observed.round(), binary_results)

        output['f1'] = metrics.f1_score(observed.round(), binary_results)

        output['auc'] = metrics.roc_auc_score(observed.round(), binary_results)

        return output