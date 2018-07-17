#!numerai/bin/python


import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, linear_model, svm, metrics, ensemble, naive_bayes
from sklearn.model_selection import ShuffleSplit

class DataLoader():

    training_data_file = 'numerai_training_data.csv'
    test_data_file = 'numerai_tournament_data.csv'

    def __init__(self, loc, date) :
        self.loc = loc + "numerai_datasets_" + date + "/"
        self.date = date

    def read(self):
        train = pd.read_csv(self.loc + self.training_data_file, header = 0)
        test = pd.read_csv(self.loc + self.test_data_file, header = 0)
        return train, test

def predictData(competitionType):

    print("Loading data...")

    # Load the data from the CSV files
    dl = DataLoader("datasets/", "17_07_2018")
    training_data, prediction_data = dl.read()

    features = [f for f in list(training_data) if "feature" in f]
    Y = training_data['target_' + competitionType]
    X = training_data[features]

#     for i in {'id', 'era', 'data_type', 'target'}:
#         X = X.drop(i, axis=1)

    models = {'logistic' : linear_model.LogisticRegression(n_jobs=1),
                  'naiveBayes' : naive_bayes.GaussianNB(),
                  'randomForest' : ensemble.RandomForestClassifier(),
                  'extraTrees' : ensemble.ExtraTreesClassifier(),
                  'gradientBoosting' : ensemble.GradientBoostingClassifier(max_depth= 20),
                  'adaBoost' : ensemble.AdaBoostClassifier()}

    model_performance = {}

    splits = 5
    ss = ShuffleSplit(n_splits=splits, test_size=0.25)
    for train_index, test_index in ss.split(training_data):
        Y_test = Y[train_index]
        X_test = X.iloc[train_index]

        for name, model in models.items():#svm.SVC(probability= True)

            print("Training " + name + "...")
            # Your model is trained on the numerai_training_data
            model.fit(X_test, Y_test)

            print("Predicting...")
            # Your trained model is now used to make predictions on the numerai_tournament_data
            # The model returns two columns: [probability of 0, probability of 1]
            # We are just interested in the probability that the target is 1.
            y_prediction = model.predict_proba(X.iloc[test_index])
            results = y_prediction[:, 1]

            if not name in model_performance:
                model_performance[name] = []

            model_performance[name].append(metrics.log_loss(Y[test_index], results))


            print(np.mean(np.array(model_performance[name])))
            print("Writing predictions to predictions.csv")


            # Save the predictions out to a CSV file

    best_model = ""
    best_acc = 0

    for mod_name, acc in model_performance.iteritems():
        mean = np.mean(np.array(acc))
        if mean > best_acc:
            best_acc = mean
            best_model = mod_name

    final_model = models[best_model].fit(X_test,Y_test)
    results = final_model.predict_proba(prediction_data[features])[:,1]

    print("The final model chosen is " + best_model + " with accuracy " + str(best_acc))

    results_df = pd.DataFrame(data={'probability': results})
    joined = pd.DataFrame(prediction_data["id"]).join(results_df)
    joined.to_csv("~/predictions.csv", index=False)
    # Now you can upload these predictions on numer.ai



if __name__ == '__main__':
    predictData('bernie')
