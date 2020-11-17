from sklearn import linear_model, ensemble, naive_bayes, metrics
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit
from sklearn.base import clone

from xgboost import XGBClassifier, XGBRegressor
from .DNN import DNNVanilla

from scipy import stats


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

        self.models['xgboost'] = XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators=self.n_est,
                                               n_jobs=-1)  # , early_stopping_rounds = 5),
        self.models['xgboost_num'] = XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=self.n_est, n_jobs=-1,
                                                  colsample_bytree=0.1)
        self.models['xgboostReg'] = XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=self.n_est,
                                                 n_jobs=-1)  # , early_stopping_rounds = 5)

        self.models['adaBoost'] = ensemble.AdaBoostClassifier()

        self.models['DNN'] = DNNVanilla(width=10, depth=1)
        self.models['DNN_full'] = DNNVanilla(width=10, depth=1)

        self.predict_only = ['xgboostReg', 'DNN']
        self.use_original_features = ['xgboost_num', 'DNN_full']

    def cross_validate_model_params(self, data, splits=10, n_cores=-1):

        parameters = {'n_estimators': stats.randint(150, 500),
                      'learning_rate': stats.uniform(0.01, 0.07),
                      'subsample': stats.uniform(0.3, 0.7),
                      'max_depth': [3, 4, 5, 6, 7, 8, 9],
                      'colsample_bytree': stats.uniform(0.5, 0.45),
                      'min_child_weight': [1, 2, 3]
                      }

        cv = ShuffleSplit(n_splits=splits)

        gscv = RandomizedSearchCV(XGBRegressor(), param_distributions=parameters, n_jobs=n_cores,
                                  scoring=metrics.make_scorer(metrics.mean_absolute_error),
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

        elif model_name in self.use_original_features:

            model.fit(train_data.getX(data_type=data_type_train, original_features=True),
                      train_data.getY(data_type=data_type_train))

            results = model.predict(test_data.getX(data_type=data_type_test, original_features=True))

        else:

            model.fit(train_data.getX(data_type=data_type_train),
                      train_data.getY(data_type=data_type_train).round())

            y_prediction = model.predict_proba(test_data.getX(data_type=data_type_test))

            results = y_prediction[:, 1]

        return results

    def __getitem__(self, model):
        return clone(self.models[model])

