from sklearn import linear_model, ensemble, naive_bayes, metrics
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge

import numpy as np

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

    predict_only = ['xgboostReg', 'xgboostReg_cv_param', 'DNN']
    use_original_features = ['xgboost_num', 'DNN_full', 'Ridge']
    unravel = ['PLSReg', 'PLSReg_cv']
    saved_model = ['xgbreg_costly']

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
        # self.models['xgboost_classifier'] = XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators=self.n_est, n_jobs=-1,
        #                                           colsample_bytree=0.1, use_label_encoder=False)
        self.models['xgboostReg'] = XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=self.n_est,
                                                 n_jobs=-1)  # , early_stopping_rounds = 5)

        self.models['adaBoost'] = ensemble.AdaBoostClassifier()

        # self.models['DNN'] = DNNVanilla(width=10, depth=10)
        # self.models['DNN_full'] = DNNVanilla(width=10, depth=10)

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

        dnn_parameters = {'width': [i for i in range(20)],
                          'depth': [i for i in range(20)],
                          'epochs': [i for i in range(20)]}


        pls_parameters = {'n_components': stats.randint(1, np.min(data.getX(original_features=True).shape))}

        gscv_pls = RandomizedSearchCV(PLSRegression(), param_distributions=pls_parameters, n_jobs=n_cores,
                                      scoring=metrics.make_scorer(metrics.mean_absolute_error),
                                      cv=cv, verbose=5, return_train_score=True)

        gscv_pls.fit(data.getX( original_features=True),
                     data.getY())

        self.models['PLSReg_cv'] = PLSRegression(**gscv_pls.best_params_)



        gscv_dnn = RandomizedSearchCV(DNNVanilla(), param_distributions=dnn_parameters, n_jobs=1,
                                      scoring=metrics.make_scorer(metrics.mean_absolute_error),
                                      cv=cv, verbose=5, return_train_score=True)

        gscv_dnn.fit(data.getX(),
                     data.getY())

        self.models['DNN'] = DNNVanilla(**gscv_dnn.best_params_)
        self.models['DNN_full'] = DNNVanilla(**gscv_dnn.best_params_)

        gscv = RandomizedSearchCV(XGBRegressor(), param_distributions=parameters, n_jobs=n_cores,
                                  # scoring=metrics.make_scorer(metrics.mean_absolute_error, greater_is_better = False),
                                  scoring=metrics.make_scorer(correlation),
                                  cv=cv, verbose=5, return_train_score=True)

        gscv.fit(data.getX(),
                 data.getY())

        self.models['xgboostReg_cv_param'] = XGBRegressor(**gscv.best_params_)


    @classmethod
    def estimate_model(cls, model, model_name, train_data,
                       test_data=None, data_type_train=None, data_type_test=None,
                       return_model = False, predict_only = False):

        if test_data is None:
            test_data = train_data

        if model_name in cls.predict_only:

            if not predict_only:
                model.fit(train_data.getX(data_type=data_type_train),
                      train_data.getY(data_type=data_type_train))

            if not return_model:
                results = model.predict(test_data.getX(data_type=data_type_test))

        elif model_name in cls.unravel:

            if not predict_only:
                model.fit(train_data.getX(data_type=data_type_train, original_features=True),
                      train_data.getY(data_type=data_type_train))

            if not return_model:
                results = model.predict(test_data.getX(data_type=data_type_test, original_features=True))
                results = results[:,0]

        elif model_name in cls.use_original_features:

            if not predict_only:
                model.fit(train_data.getX(data_type=data_type_train, original_features=True),
                      train_data.getY(data_type=data_type_train))

            if not return_model:
                results = model.predict(test_data.getX(data_type=data_type_test, original_features=True))

        elif model_name in cls.saved_model:

            if not return_model:
                results = model.predict(test_data.getX(data_type=data_type_test, original_features=True))

        else:

            if not predict_only:
                model.fit(train_data.getX(data_type=data_type_train),
                      train_data.getY(data_type=data_type_train).round())

            if not return_model:
                y_prediction = model.predict_proba(test_data.getX(data_type=data_type_test))
                results = y_prediction[:, 1]

        if return_model:
            return model

        else:
            return results

    def __getitem__(self, model):
        if model == 'xgbreg_costly':
            return self.xgboost_costly
        else:
            return clone(self.models[model])
