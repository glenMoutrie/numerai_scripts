#!/usr/bin/env python

"""
Example classifier on Numerai data using a logistic regression classifier.
To get started, install the required packages: pip install pandas, numpy, sklearn
"""

import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, linear_model, KFold, svm, ShuffleSplit

def trainModels(X_train, Y_train, X_test, Y_test):
    log_reg = linear_model.LogisticRegression(n_jobs = 1)
    svm_model = svm.SVC()

    log_reg.fit(X_train,y_train)
    svm_model.fit(X_train,y_train)

    log_pred = log_reg.predict_proba(X_test)
    svm_pred = svm_model(X_test)




def main():
    # Set seed for reproducibility
    np.random.seed(0)

    print("Loading data...")
    # Load the data from the CSV files
    training_data = pd.read_csv('~/Downloads/numerai_datasets 3/numerai_training_data.csv', header=0)
    prediction_data = pd.read_csv('~/Downloads/numerai_datasets 3/numerai_tournament_data.csv', header=0)

    # Transform the loaded CSV data into numpy arrays
    Y = training_data['target']
    X = training_data.drop('target', axis=1)

    ss = ShuffleSplit(n_splits = 5, test.size = 0.25)
    for train_index, test_index in ss.split(X):



    t_id = prediction_data['t_id']
    x_prediction = prediction_data.drop('t_id', axis=1)

    # This is your model that will learn to predict
    model = linear_model.LogisticRegression(n_jobs=-1)

    print("Training...")
    # Your model is trained on the numerai_training_data
    model.fit(X, Y)

    print("Predicting...")
    # Your trained model is now used to make predictions on the numerai_tournament_data
    # The model returns two columns: [probability of 0, probability of 1]
    # We are just interested in the probability that the target is 1.
    y_prediction = model.predict_proba(x_prediction)
    results = y_prediction[:, 1]
    results_df = pd.DataFrame(data={'probability':results})
    joined = pd.DataFrame(t_id).join(results_df)

    print("Writing predictions to predictions.csv")
    # Save the predictions out to a CSV file
    joined.to_csv("predictions.csv", index=False)
    # Now you can upload these predictions on numer.ai


if __name__ == '__main__':
    main()
