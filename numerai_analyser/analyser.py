#!numerai/bin/python
import pandas as pd

from .data_manager import *
from .model_automation import *
from .test_type import TestType

# Route map for model_improvements branch
# 1) Improve logging so that you can assess improvement of implementation
#       a) More measurements of predictive accuarcy
#       b) Bench mark model run times
#       c) Break down of era's with good/bad predictions
#       d) Potentially consider implementing postgres db...
#
# 2) Model selection
#       a) Create a better voting system for model selection, consider going for an ensemble approach
#       b) This needs to rely somewhat on the architecture used for logging
#       c) Better hyperparamter selection
#       d) Maybe... maybe implement dnn...
#       e) incorporate boom spike slab without polynomial
#
# 3) Performance
#       a) Better parallelisation using Dask for model estimation 
#       b) multiprocessing/dask for different cuts of the data

def predictNumerai(test_run = False, test_type = TestType.SYNTHETIC_DATA, test_size = 100):

    dl = DataLoader()

    competitions = dl.getCompetitions()

    print('Running on the following competitions:')

    print(competitions)

    if test_run:
        out = '(Test Run): '

        if test_type is TestType.SYNTHETIC_DATA:
            out += 'synthetic data test'
        elif test_type is TestType.SUBSET_DATA:
            out += 'subset data test'

        print(out)

    for comp in competitions:

        print('Running on comp ' + comp)
        
        if not test_run or test_type is TestType.SUBSET_DATA:
            dl.downloadLatest()

        dl.read(test_run, test_type, test_size)

        os.environ["OMP_NUM_THREADS"] = "8"

        train, test = dl.getData(comp)

        # train.generatePolynomialFeatures()
        # print(test)
        # test = train.setPolynomialFeatures(test)

        if test_run:
            n_est = 200
        else:
            n_est = 20000
    
        models = {
        'logistic' : linear_model.LogisticRegression(),
        'naiveBayes' : naive_bayes.GaussianNB(),
        'randomForest' : ensemble.RandomForestClassifier(),
        'extraTrees' : ensemble.ExtraTreesClassifier(),
        'gradientBoosting' : ensemble.GradientBoostingClassifier(),
        'xgboost' : XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators= n_est),
        'xgboostReg' : XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators= n_est),
        'adaBoost' : ensemble.AdaBoostClassifier()
        }

        tester = ModelTester(models, train.getEras(), 3, 0.25)

        tester.testAllSplits(train)

        results = tester.getBestPrediction(train, test)

        results_col = 'probability_' + comp

        results_df = pd.DataFrame(data={results_col: results})
        results_df = pd.DataFrame(test.getID()).join(results_df)

        # results_df[results_col].loc[results_df[results_col] > 0.7] = 0.7
        # results_df[results_col].loc[results_df[results_col] < 0.3] = 0.3

        if not test_run:

            dl.write(results_df)
            dl.uploadResults(comp)

            try:
                dl.getSubmissionStatus()
            except ValueError as error:
                print("Caught error in upload for " + comp)
                print(error)

        print("Complete.")


