#!numerai/bin/python


import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, linear_model, svm, metrics, ensemble, naive_bayes
from sklearn.model_selection import ShuffleSplit


# Plan:
# 1) create a data loader that creates a NumeraiData class
# 2) Define a NumeraiData Class with features, x data and y data
# 3) finish the model tester that returns the model results, estimating in parallel using multiprocessing
# 4) automate with numerapi and other such tools

# CURRENT TODO: FIX BUGS IN UPDATE

class DataSet():

    y_full = np.ndarray(None)
    x_full = np.ndarray(None)

    y_test_split = np.ndarray(None)
    y_train_split = np.ndarray(None)

    x_test_split = np.ndarray(None)
    x_train_split = np.ndarray(None)

    

    def __init__(self, X, Y):

        self.y_full = self.y_test_split = Y
        self.x_full = self.x_test_split = X

        self.N = Y.shape[0]
        self.split_index = {'train' : [i for i in range(0,self.N)], 'test' : []}

    def updateSplit(self, train_ind, test_ind):

        self.y_train_split = Y[train_index]
        self.x_train_split = X.iloc[train_index]

        self.x_test_split = X.iloc[test_index]
        self.y_test_split = Y[train_index]

        self.split_index = {'train' : train_index, 'test' : test_index}

    def getTrainingData(self):
        return self.y_train_split, self.x_train_split

    def getTestData(self):
        return self.y_test_split, self.x_test_split

    def getY(self, train):
        if train:
            return self.y_train_split
        else:
            return self.y_test_split

    def getX(self, train):
        if train:
            return self.x_train_split
        else:
            return self.x_test_split

    def getXFull(self):
        return self.x_full




class DataLoader():

    training_data_file = 'numerai_training_data.csv'
    test_data_file = 'numerai_tournament_data.csv'



    def __init__(self, loc, date) :
        self.loc = loc + "numerai_datasets_" + date + "/"
        self.date = date

    def read(self):
        self.train = pd.read_csv(self.loc + self.training_data_file, header = 0)
        self.test = pd.read_csv(self.loc + self.test_data_file, header = 0)

        self.features = [f for f in list(self.train) if "feature" in f]

    def write(self, output):
        output.to_csv(self.loc + "predictions.csv", index = False)

    def getData(self, competition_type):
        self.train = DataSet(self.train[self.features], self.train['target_' + competition_type])
        self.test = self.test[self.features]

        return self.train, self.test




class modelTester():

    models = {'logistic' : linear_model.LogisticRegression(n_jobs=1),
                  'naiveBayes' : naive_bayes.GaussianNB(),
                  'randomForest' : ensemble.RandomForestClassifier(),
                  'extraTrees' : ensemble.ExtraTreesClassifier(),
                  # 'gradientBoosting' : ensemble.GradientBoostingClassifier(),
                  'adaBoost' : ensemble.AdaBoostClassifier()}

    def __init__(self, splits = 5, test_size = 0.25) :
        self.splits = splits
        self.ss = ShuffleSplit(n_splits = splits, test_size = test_size)
        self.model_performance = {}


    def testAllSplits(self, data):
        for train_i, test_i in self.ss.split(data.getXFull()):
            data.updateSplit(train_i, test_i)
            __testAllModels(data)



    def testAllModels(data, models):
        print({model: testModel(data, models.get(model)) for model in models.keys()})


    def __testModel(X_train, X_test, Y_train, Y_test, model):

        model.fit(data.getX(True), data.getY(True))

        y_prediction = model.predict_proba(data.getX(False))

        results = y_prediction[:, 1]

        return metrics.log_loss(data.getY(False), results)





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
                  # 'gradientBoosting' : ensemble.GradientBoostingClassifier(),
                  'adaBoost' : ensemble.AdaBoostClassifier()}

    model_performance = {}

    # splits = 5
    splits = 1
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

    for mod_name, acc in model_performance.items():
        mean = np.mean(np.array(acc))
        if mean > best_acc:
            best_acc = mean
            best_model = mod_name

    final_model = models[best_model].fit(X_test,Y_test)
    results = final_model.predict_proba(prediction_data[features])[:,1]

    print("The final model chosen is " + best_model + " with accuracy " + str(best_acc))

    results_df = pd.DataFrame(data={'probability': results})
    joined = pd.DataFrame(prediction_data["id"]).join(results_df)
    dl.write(joined)
    # Now you can upload these predictions on numer.ai



if __name__ == '__main__':

    dl = DataLoader("datasets/", "17_07_2018")
    dl.read()
    
    train, test = dl.getData('bernie')
    
    tester = modelTester(1, 0.25)

    tester.testAllSplits(train)


    # predictData('bernie')
