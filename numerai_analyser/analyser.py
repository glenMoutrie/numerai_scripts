#!numerai/bin/python
import pandas as pd

from .model_automation import *
from .config import NumeraiConfig
from .data_manager import NumeraiDataManager
from .test_type import TestType

# Route map for model_improvements branch
# 1) Improve logging so that you can assess improvement of implementation
#       a) More measurements of predictive accuarcy - DONE
#       b) Bench mark model run times - DONE
#       c) Break down of era's with good/bad predictions - DONE
#       d) Potentially consider implementing postgres db...
#
# 2) Model selection
#       a) Create a better voting system for model selection, consider going for an ensemble approach - DONE
#       b) This needs to rely somewhat on the architecture used for logging - DONE
#       c) Better hyperparamter selection - DONE(ish)
#       d) Maybe... maybe implement dnn... - DONE
#       e) incorporate boom spike slab without polynomial - DONE
#
# 3) Performance
#       a) Better parallelisation using Dask for model estimation 
#       b) multiprocessing/dask for different cuts of the data

def predictNumerai(test_run = False, test_type = TestType.SYNTHETIC_DATA, test_size = 100):

    config = NumeraiConfig(test_run, test_type)

    config.setup()

    dl = NumeraiDataManager(config)

    competitions = dl.getCompetitions()

    config.logger.info('Running on the following competitions: ' + ', '.join(competitions))

    for comp in competitions:

        config.logger.info('Running on comp ' + comp)
        
        if not test_run or test_type is TestType.SUBSET_DATA:
            dl.downloadLatest()

        dl.read(test_run, test_type, test_size)

        train, test = dl.getData(comp, True, True, test_run)

        if test_run:
            n_est = 200
        else:
            # n_est = 20000
            n_est = 200
    
        models = {
        'logistic' : linear_model.LogisticRegression(),
        'naiveBayes' : naive_bayes.GaussianNB(),
        'randomForest' : ensemble.RandomForestClassifier(),
        'extraTrees' : ensemble.ExtraTreesClassifier(),
        'gradientBoosting' : ensemble.GradientBoostingClassifier(),
        'xgboost' : XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators = n_est),#, early_stopping_rounds = 5),
        'xgboost_num': XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators= 20000, n_jobs=-1, colsample_bytree=0.1),
        'xgboostReg' : XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators = n_est),#, early_stopping_rounds = 5),
        'adaBoost' : ensemble.AdaBoostClassifier(),
        'DNN': DNNVanilla(width = 10, depth = 1)
        }

        tester = ModelTester(models, train.getEras(), config, 3, 0.25)

        tester.testAllSplits(train)

        results = tester.getBestPrediction(train, test)

        results_col = 'probability_' + comp

        results_df = pd.DataFrame(data={results_col: results})
        results_df = pd.DataFrame(test.getID()).join(results_df)

        if not test_run:

            dl.uploadResults(results_df, comp)

            try:
                dl.getSubmissionStatus()
            except ValueError as error:
                config.logger.error("Caught error in upload for " + comp)
                config.logger.error(error)

        config.logger.info("Complete.")



