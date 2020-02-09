import os
import pandas as pd
from itertools import compress

class FeatureSelection:

    def __init__(self, data, features, target, use_previous = False):

        self.wd = os.getcwd()
        self.file_location = self.wd + "/.temp/num_data.csv"
        self.output_location = self.wd + "/.temp/output"
        
        self.original_features = features


        if not use_previous:
            self.constructFormula(features, target)
            self.writeDataToTempFile(data)
            self.executeBayesianFeatureSelectionR()

        self.output = pd.read_csv(self.output_location)


    def executeBayesianFeatureSelectionR(self):
        
        command = "Rscript r_scripts/feature_select.R"
        command += " " + self.file_location + " "
        command += " " + self.formula + " "
        command += " " + self.output_location

        os.system(command)
    

    def writeDataToTempFile(self, data):
        if not ".temp" in os.listdir():
            os.mkdir(".temp")

        data.to_csv(self.file_location, index = False)

    def constructFormula(self, features, target) :

        formula = "'" + target + " ~ "
        n = len(features)

        for i in range(0, n):
            formula += features[i]
            if not i == (n - 1):
                formula += " + "

        formula += "'"
        self.formula = formula

    def selectBestFeatures(self,min_include, exclude_intercept = True):

        if self.output.loc[0]['variable'] == "NA":

            self.best_vars = self.original_features
            return(self.best_vars)


        best_vars = list(self.output.loc[self.output['probability'] > min_include]['variable'])

        if len(best_vars) < 1:
            print("No features found with good predictive probability")
            best_vars = list(self.output.loc[self.output['probability'] == max(self.output['probability'])]['variable'])


        if exclude_intercept:
            best_vars = list(compress(best_vars, list(map(lambda x: x != '(Intercept)', best_vars))))

        self.best_vars = best_vars

        return(self.best_vars)



if __name__ == "__main__":
    
    test = pd.DataFrame({"one" : [1,2,3], "two": [4,5,6], "pred" : [0, 0, 1]})
    fs = FeatureSelection(test, ["one", "two"], "pred")

    print(fs.output)
    print(fs.output.iloc[[1],[0]])
    print(fs.selectBestFeatures(0.1))
    # fs.getFeatures()
    # fs.getInclusionProbability()

