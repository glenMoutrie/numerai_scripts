from numerai_analyser.config import NumeraiConfig
from numerai_analyser.data_manager import NumeraiDataManager
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
# from sklearn.metrics import mean_absolute_error
from dask_ml.metrics import mean_absolute_error
from apricot import FeatureBasedSelection
from sklearn.pipeline import Pipeline
import umap
import pandas as pd
# from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# Import the datasets

training_data = pd.read_csv("datasets/numerai_dataset_217/numerai_training_data.csv")
test_data = pd.read_csv("datasets/numerai_dataset_217/numerai_tournament_data.csv")
submission = pd.read_csv("datasets/numerai_dataset_217/2020_06_21_113628_predictions.csv")

features = [i for i in test_data.columns if i.startswith("feature_")]
target = "target_kazutsugi"


# Performing UMAP and clustering

sampler = FeatureBasedSelection(500)

sub_sample = sampler.fit_transform(training_data[features].to_numpy())

n_comp = 5

print("Estimating UMAP")
reducer = umap.UMAP(n_components = n_comp)
reducer.fit(sub_sample)

dim_names = ["dim_" + str(i) for i in range(n_comp)]

X = reducer.transform(training_data[features])

umap_df = pd.DataFrame(X)\
.rename(columns = dict(zip([i for i in range(n_comp)], dim_names)))

# scatter_plt = umap_df\
# .plot.scatter('one', 'two',
#               figsize = (20,20),
#               s = 0.001 + (0.01 * training_data.target_kazutsugi),
#               c = training_data\
#               .target_kazutsugi\
#               .apply(lambda x: {0:'white', 0.25: 'white', 0.5:'white', 0.75:'white', 1:'black'}[x]))
#
# plt.show()

mod = KMeans(5)
mod.fit(umap_df[dim_names])
mod.predict(umap_df[dim_names])

umap_df['cluster'] = mod.predict(umap_df[dim_names])
umap_df['target'] = training_data.target_kazutsugi

umap_df.groupby('cluster').target.value_counts(normalize = True)

training_data['cluster'] = pd.Categorical(umap_df.cluster)

umap_features = dim_names + ['cluster']

# XGBoost parameterisation and prediction

from sklearn.model_selection import RandomizedSearchCV, GroupShuffleSplit
from xgboost import XGBRegressor
from sklearn import metrics
from scipy import stats

parameters = {'n_estimators': stats.randint(150, 500),
              'learning_rate': stats.uniform(0.01, 0.07),
              'subsample': stats.uniform(0.3, 0.7),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.45),
              'min_child_weight': [1, 2, 3]
             }

scores = {'mae' : metrics.make_scorer(metrics.mean_absolute_error),
          'explained_variance': metrics.make_scorer(metrics.explained_variance_score)}

cv = GroupShuffleSplit()

gscv = RandomizedSearchCV(XGBRegressor(), param_distributions=parameters, n_jobs = 7,
                          scoring= metrics.make_scorer(metrics.mean_absolute_error),
                          cv= cv, verbose = 5, return_train_score=True)


# gscv.fit(training_data[training_data.era.isin(['era1', 'era2', 'era3'])][features],
#          training_data[training_data.era.isin(['era1', 'era2', 'era3'])][target],
#          training_data[training_data.era.isin(['era1', 'era2', 'era3'])].era)

gscv.fit(umap_df[umap_features],
         training_data[target],
         training_data.era)

gscv.best_estimator_.get_params()

X_test = reducer.transform(test_data[features])
X_test = pd.DataFrame(X_test)\
.rename(columns = dict(zip([i for i in range(n_comp)], dim_names)))

X_test['cluster'] = mod.predict(X_test[dim_names])


predictions = gscv.predict(X_test[umap_features])
predictions = pd.Series(predictions)

import numpy as np

def score(preds, truth):
    # method="first" breaks ties based on order in array
    return np.corrcoef(truth, preds.rank(pct=True, method="first"))[0, 1]

test_data['prediction'] = predictions
test_data['submission'] = submission.probability_kazutsugi

test_data[test_data.data_type == 'validation'].groupby('era').apply(lambda x: score(x.target_kazutsugi, x.prediction)).agg(['mean', 'std'])
test_data[test_data.data_type == 'validation'].groupby('era').apply(lambda x: score(x.target_kazutsugi, x.submission)).agg(['mean', 'std'])

print(score(predictions[test_data.data_type == 'validation'], test_data[test_data.data_type == 'validation'][target]))
print(score(submission.probability_kazutsugi[test_data.data_type == 'validation'], test_data[test_data.data_type == 'validation'][target]))