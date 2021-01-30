from sklearn import linear_model, ensemble, naive_bayes, metrics
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge

import numpy as np

import pandas as pd

from xgboost import XGBClassifier, XGBRegressor
from .DNN import DNNVanilla

from scipy import stats

# Submissions are scored by spearman correlation
def correlation(predictions, targets):
    ranked_preds = predictions.rank(pct=True, method="first")
    return np.corrcoef(ranked_preds, targets)[0, 1]


# convenience method for scoring
def score(df):
    return correlation(df[PREDICTION_NAME], df[TARGET_NAME])

class ModelFactory:

    def __init__(self, n_est=200):

        self.n_est = n_est

        self.models = {}

        self._create_model_classes()

    def _create_model_classes(self):

        self.models['logistic'] = linear_model.LogisticRegression()
        self.models['naiveBayes'] = naive_bayes.GaussianNB()
        self.models['randomForest'] = ensemble.RandomForestClassifier()
        self.models['extraTrees'] = ensemble.ExtraTreesClassifier()
        self.models['gradientBoosting'] = ensemble.GradientBoostingClassifier()
        self.models['PLSReg'] = PLSRegression()
        self.models['Ridge'] = Ridge()

        self.models['xgboost_num'] = XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=self.n_est, n_jobs=-1,
                                                  colsample_bytree=0.1)
        self.models['xgboostReg'] = XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=self.n_est,
                                                 n_jobs=-1)  # , early_stopping_rounds = 5)

        self.models['adaBoost'] = ensemble.AdaBoostClassifier()

        self.models['DNN'] = DNNVanilla(width=10, depth=10)
        self.models['DNN_full'] = DNNVanilla(width=10, depth=10)

        self.predict_only = ['xgboostReg', 'DNN']
        self.use_original_features = ['xgboost_num', 'DNN_full', 'Ridge']
        self.unravel = ['PLSReg']
        self.saved_model = ['xgbreg_costly']

    def estimate_costly_models(self, data):
        self.models['xgbreg_costly'] = XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=2500, n_jobs=-1,
                                            colsample_bytree=0.1)
        self.xgboost_costly = self.models['xgbreg_costly'].fit(data.getX(original_features=True),
                                         data.getY())


    def cross_validate_model_params(self, data, splits=10, n_cores=-1):

        parameters = {'n_estimators': stats.randint(150, 1000),
                      # 'n_estimators': stats.randint(150, 500),
                      'learning_rate': stats.uniform(0.01, 0.07),
                      'subsample': stats.uniform(0.3, 0.7),
                      'max_depth': [3, 4, 5, 6, 7, 8, 9],
                      'colsample_bytree': stats.uniform(0.5, 0.45),
                      'min_child_weight': [1, 2, 3]
                      }
        cv = ShuffleSplit(n_splits=splits)

        # dnn_parameters = {'width': [i for i in range(20)],
        #                   'depth': [i for i in range(20)]}
        #
        #
        #
        # gscv_dnn = RandomizedSearchCV(DNNVanilla(), param_distributions=dnn_parameters, n_jobs=n_cores,
        #                               scoring=metrics.make_scorer(metrics.mean_absolute_error),
        #                               cv=cv, verbose=5, return_train_score=True)
        #
        # gscv_dnn.fit(data.getX(),
        #              data.getY())
        #
        # self.models['DNN'] = DNNVanilla(**gscv.best_params_)
        # self.models['DNN_full'] = DNNVanilla(**gscv.best_params_)

        gscv = RandomizedSearchCV(XGBRegressor(), param_distributions=parameters, n_jobs=n_cores,
                                  # scoring=metrics.make_scorer(metrics.mean_absolute_error, greater_is_better = False),
                                  scoring=metrics.make_scorer(correlation),
                                  cv=cv, verbose=5, return_train_score=True)

        gscv.fit(data.getX(),
                 data.getY())

        self.models['xgboostReg_cv_param'] = XGBRegressor(**gscv.best_params_)
        self.predict_only.append('xgboostReg_cv_param')



    def estimate_model(self, model_name, train_data,
                       test_data=None, data_type_train=None, data_type_test=None):

        if test_data is None:
            test_data = train_data

        model = self[model_name]

        if model_name in self.predict_only:

            model.fit(train_data.getX(data_type=data_type_train),
                      train_data.getY(data_type=data_type_train))

            results = model.predict(test_data.getX(data_type=data_type_test))

        elif model_name in self.unravel:

            model.fit(train_data.getX(data_type=data_type_train, original_features=True),
                      train_data.getY(data_type=data_type_train))

            results = model.predict(test_data.getX(data_type=data_type_test, original_features=True))
            results = results[:,0]

        elif model_name in self.use_original_features:

            model.fit(train_data.getX(data_type=data_type_train, original_features=True),
                      train_data.getY(data_type=data_type_train))

            results = model.predict(test_data.getX(data_type=data_type_test, original_features=True))

        elif model_name in self.saved_model:

            model = self.xgboost_costly
            results = model.predict(test_data.getX(data_type=data_type_test, original_features=True))

        else:

            model.fit(train_data.getX(data_type=data_type_train),
                      train_data.getY(data_type=data_type_train).round())

            y_prediction = model.predict_proba(test_data.getX(data_type=data_type_test))

            results = y_prediction[:, 1]

        # results = pd.DataFrame(results)

        return results

    def __getitem__(self, model):
        return clone(self.models[model])

