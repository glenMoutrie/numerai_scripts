#!numerai/bin/python
import pandas as pd
from data_manager import *
from model_automation import *


# Plan:
# 1) create a data loader that creates a NumeraiData class
# 2) Define a NumeraiData Class with features, x data and y data
# 3) finish the model tester that returns the model results, estimating in parallel using multiprocessing
# 4) automate with numerapi and other such tools


def predictNumerai():
    dl = DataLoader()
    dl.downloadLatest()
    dl.read()
    
    comp = 'bernie'

    train, test = dl.getData(comp)
    # train.generatePolynomialFeatures()
    # print(test)
    # test = train.setPolynomialFeatures(test)
    
    tester = ModelTester(5, 0.25)

    tester.testAllSplits(train)

    results = tester.getBestPrediction(train, test)

    results_df = pd.DataFrame(data={'probability_'+comp: results})
    results_df = pd.DataFrame(test.getID()).join(results_df)

    dl.write(results_df)
    dl.uploadResults(comp)
    dl.getSubmissionStatus()

    print("Complete.")

if __name__ == '__main__':
    predictNumerai()

