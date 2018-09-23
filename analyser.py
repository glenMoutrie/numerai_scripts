#!numerai/bin/python
import pandas as pd
from data_manager import *
from model_automation import *


# Plan:
# 1) create a data loader that creates a NumeraiData class
# 2) Define a NumeraiData Class with features, x data and y data
# 3) finish the model tester that returns the model results, estimating in parallel using multiprocessing
# 4) automate with numerapi and other such tools


if __name__ == '__main__':

    dl = DataLoader("datasets/", "17_07_2018")
    dl.read()
    
    train, test = dl.getData('bernie')
    
    tester = ModelTester(5, 0.25)

    tester.testAllSplits(train)

    results = tester.getBestPrediction(train, test[dl.features])

    results_df = pd.DataFrame(data={'probability': results})
    results_df = pd.DataFrame(test["id"]).join(results_df)
    
    dl.write(results_df)




    # predictData('bernie')
