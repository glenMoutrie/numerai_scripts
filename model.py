#!numerai/bin/python


import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, linear_model, svm, metrics
from sklearn.model_selection import ShuffleSplit

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
    training_data = pd.read_csv('~/Downloads/numerai_datasets/numerai_training_data.csv', header=0)
    prediction_data = pd.read_csv('~/Downloads/numerai_datasets/numerai_tournament_data.csv', header=0)

    Y = training_data['target']
    X = training_data
    for i in {'id', 'era', 'data_type','target'}:
            X = X.drop(i, axis=1)

    ss = ShuffleSplit(n_splits = 5, test_size = 0.25)
    for train_index, test_index in ss.split(training_data):

        Y_test = Y[train_index]
        X_test = X.iloc[train_index]

        # This is your model that will learn to predict
        model = linear_model.LogisticRegression(n_jobs=-1)

        print("Training...")
        # Your model is trained on the numerai_training_data
        model.fit(X_test, Y_test)

        print("Predicting...")
        # Your trained model is now used to make predictions on the numerai_tournament_data
        # The model returns two columns: [probability of 0, probability of 1]
        # We are just interested in the probability that the target is 1.
        y_prediction = model.predict_proba(X.iloc[test_index])
        results = y_prediction[:, 1]

        results_df = pd.DataFrame(data={'probability':results})
        print(metrics.log_loss(Y[test_index],results))
        print("Writing predictions to predictions.csv")
    # Save the predictions out to a CSV file
    joined.to_csv("predictions.csv", index=False)
    # Now you can upload these predictions on numer.ai

# def main():
#     # Set seed for reproducibility
#     np.random.seed(0)

#     print("Loading data...")
#     # Load the data from the CSV files
#     training_data = pd.read_csv('~/Downloads/numerai_datasets/numerai_training_data.csv', header=0)
#     prediction_data = pd.read_csv('~/Downloads/numerai_datasets/numerai_tournament_data.csv', header=0)


#     # Transform the loaded CSV data into numpy arrays
#     features = [f for f in list(training_data) if "feature" in f]
#     X = training_data[features]
#     Y = training_data["target"]
#     x_prediction = prediction_data[features]
#     ids = prediction_data["id"]

#     # This is your model that will learn to predict
#     model = linear_model.LogisticRegression(n_jobs=-1)

#     print("Training...")
#     # Your model is trained on the training_data
#     model.fit(X, Y)

#     print("Predicting...")
#     # Your trained model is now used to make predictions on the numerai_tournament_data
#     # The model returns two columns: [probability of 0, probability of 1]
#     # We are just interested in the probability that the target is 1.
#     y_prediction = model.predict_proba(x_prediction)
#     results = y_prediction[:, 1]
#     results_df = pd.DataFrame(data={'probability':results})
#     joined = pd.DataFrame(ids).join(results_df)

#     print("Writing predictions to predictions.csv")
#     # Save the predictions out to a CSV file
#     joined.to_csv("predictions.csv", index=False)
#     # Now you can upload these predictions on numer.ai



if __name__ == '__main__':
    main()
