from sklearn import linear_model, ensemble, naive_bayes, metrics
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit

from xgboost import XGBClassifier, XGBRegressor
from .DNN import DNNVanilla

from scipy import stats


class ModelFactory:

    def __init__(self, n_est = 200):

        self.n_est = n_est

        self.models = {}

        self.createModelClasses()

    def createModelClasses(self):

        self.models['logistic'] = linear_model.LogisticRegression()
        self.models['naiveBayes'] = naive_bayes.GaussianNB()
        self.models['randomForest'] = ensemble.RandomForestClassifier()
        self.models['extraTrees'] = ensemble.ExtraTreesClassifier()
        self.models['gradientBoosting'] = ensemble.GradientBoostingClassifier()

        self.models['xgboost'] = XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators= self.n_est, n_jobs = -1)  # , early_stopping_rounds = 5),
        self.models['xgboost_num'] = XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators= self.n_est, n_jobs=-1,
                                    colsample_bytree=0.1)
        self.models['xgboostReg'] = XGBRegressor(max_depth=5, learning_rate=0.01,n_estimators= self.n_est,n_jobs = -1)  # , early_stopping_rounds = 5)

        self.models['adaBoost'] = ensemble.AdaBoostClassifier()

        self.models['DNN'] = DNNVanilla(width=10, depth=1)
        self.models['DNN_full'] = DNNVanilla(width=10, depth=1)

        self.predict_only = ['xgboostReg', 'DNN']
        self.use_all_features = ['xgboost_num', 'DNN_full']

    def cross_validate_model_params(self, data, splits = 10):

        parameters = {'n_estimators': stats.randint(150, 500),
                      'learning_rate': stats.uniform(0.01, 0.07),
                      'subsample': stats.uniform(0.3, 0.7),
                      'max_depth': [3, 4, 5, 6, 7, 8, 9],
                      'colsample_bytree': stats.uniform(0.5, 0.45),
                      'min_child_weight': [1, 2, 3]
                      }

        cv = ShuffleSplit(n_splits=splits)

        gscv = RandomizedSearchCV(XGBRegressor(), param_distributions=parameters, n_jobs=7,
                                  scoring=metrics.make_scorer(metrics.mean_absolute_error),
                                  cv=cv, verbose=5, return_train_score=True)

        gscv.fit(data.getX(),
                 data.getY())

        self.models['xgboostReg_cv_param'] = XGBRegressor(**gscv.best_params_)
        self.predict_only.append('xgboostReg_cv_param')

