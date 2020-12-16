import pandas as pd
import numpy as np
import scipy as sp

from joblib import Parallel, delayed

import logging

from .model_metrics import ModelMetrics

def _min_max(x):
    return (x - min(x))/(max(x) - min(x))

def _neutralize(df, columns, by, proportion=1.0):
    scores = df[columns]
    exposures = df[by].values
    scores = scores - proportion * exposures.dot(np.linalg.pinv(exposures).dot(scores))

    scores = scores / scores.std()

    scores[scores < 0] = 0
    scores[scores > 1] = 1

    return scores


def _normalize(df):
    X = (df.rank(method="first") - 0.5) / len(df)
    return sp.stats.norm.ppf(X)


def _normalize_and_neutralize(df, columns, by, proportion=1.0):
    # Convert the scores to a normal distribution
    df.loc[:,columns] = _normalize(df[columns])
    df.loc[:,columns] = _neutralize(df, columns, by, proportion)
    return df[columns]


def normalizeAndNeutralize(predictions, test, proportion=1.0):
    X = test.getX(data_type=None, all_features=True)

    output = X \
        .assign(PREDS=predictions,
                ERA=test.full_set.era) \
        .groupby('ERA') \
        .apply(lambda x: _normalize_and_neutralize(x, ["PREDS"], test.all_features, proportion))['PREDS']

    return {'proportion': proportion, 'neut_pred': output}


def auto_neutralize_normalize(predictions, test, cores=-1, logger=logging.getLogger('neutralize-normalize')):
    parallel = Parallel(n_jobs=cores, require='sharedmem')

    logger.info('Selecting neutralization proportion')

    neutralized = parallel(delayed(normalizeAndNeutralize)(predictions, test, i / 10) for i in range(11))

    scores = [{**{'proportion': n['proportion']}, \
               **ModelMetrics.getNumeraiScoreByEra(test.getY(), n['neut_pred'], test.getEras())}
              for n in neutralized]

    scores = pd.DataFrame(scores)

    best_prop = scores \
        .iloc[scores[['correlation', 'sharpe']] \
        .apply(_min_max) \
        .apply(sum, axis=1) \
        .idxmax()]

    logger.info('Neutralization proportion {0} with corr {1:.2f} and sharpe {2:.2f}'.format(best_prop.proportion,
                                                                                            best_prop.correlation,
                                                                                            best_prop.sharpe))

    return pd.DataFrame(neutralized) \
        .set_index('proportion') \
        .loc[best_prop.proportion][0]

