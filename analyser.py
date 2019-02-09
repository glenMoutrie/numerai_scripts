#!numerai/bin/python
import pandas as pd
from data_manager import *
from model_automation import *


# Plan:
# 1) create a data loader that creates a NumeraiData class -- DONE
# 2) Define a NumeraiData Class with features, x data and y data -- DONE
# 3) finish the model tester that returns the model results, estimating in parallel using multiprocessing
# 4) automate with numerapi and other such tools -- DONE
# 5) Better predictive models, look at alternaitves, betters model specifications
# 6) Automatic feature selection --DONE
# 7) Feature engineering (look at clustering etc)
# 8) Fix the cross terms issue by removing white space in names
# 9) Potentially try a deep learning approach...
# 10) Try an ensemble approach accross different epochs


def predictNumerai(test_run = False):

    # for comp in ['frank', 'hillary']:
    for comp in ['bernie','elizabeth', 'jordan', 'ken', 'charles', 'frank', 'hillary']:

        print('Running on comp ' + comp)
        dl = DataLoader()

        if not test_run:
            dl.downloadLatest()

        dl.read(test_run)

        os.environ["OMP_NUM_THREADS"] = "8"

        train, test = dl.getData(comp)
        train.generatePolynomialFeatures()
        print(test)
        test = train.setPolynomialFeatures(test)
    

        models = {
    # 'logistic' : linear_model.LogisticRegression(),
    #               'naiveBayes' : naive_bayes.GaussianNB(),
    #               'randomForest' : ensemble.RandomForestClassifier(),
    #               'extraTrees' : ensemble.ExtraTreesClassifier(),
    #               'gradientBoosting' : ensemble.GradientBoostingClassifier(),
                  'xgboost' : XGBClassifier()
                  # 'adaBoost' : ensemble.AdaBoostClassifier()
                  }

        tester = ModelTester(models, 1, 0.25)

        tester.testAllSplits(train)

        results = tester.getBestPrediction(train, test)

        results_col = 'probability_'+comp

        results_df = pd.DataFrame(data={results_col: results})
        results_df = pd.DataFrame(test.getID()).join(results_df)

        results_df[results_col].loc[results_df[results_col] > 0.7] = 0.7
        results_df[results_col].loc[results_df[results_col] < 0.3] = 0.3

        dl.write(results_df)

        if not test_run:
            dl.uploadResults(comp)

        try:
            dl.getSubmissionStatus()
        except ValueError as error:
            print("Caught error in upload for " + comp)
            print(error)

        print("Complete.")

if __name__ == '__main__':
    predictNumerai(True)

