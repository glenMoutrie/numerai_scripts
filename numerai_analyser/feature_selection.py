from itertools import compress

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


class FeatureSelection:

    """
    Feature Selection using BoomSpikeSlab from R

    This class invokes an R instance using rpy2, and calls BoomSpikeSlab to estimate the probability that each
    columns should be used.
    """

    def __init__(self, data, features, target):

        self.original_features = features
        self.constructFormula(features, target)
        self.output = self.runLMSpike(data)

    def runLMSpike(self, data):

        importr('BoomSpikeSlab')

        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(data)

        ro.r("""
        getProbabilities <- function(form, data) {
    
            form <- formula(form)
    
            mm <- model.matrix(form, data)
            resp <- model.response(model.frame(form, data))
    
            model <- lm.spike(form, niter=500, data)
            output <- colMeans(model$beta != 0)
    
            output <- data.frame(variable=names(output), probability=output)
    
            output
    
        }
        """)

        results = ro.globalenv['getProbabilities'](self.formula, r_df)

        with localconverter(ro.default_converter + pandas2ri.converter):
            results = ro.conversion.rpy2py(results)

        results = results.reset_index(drop=True)

        return (results)

    def constructFormula(self, features, target):

        formula = target + " ~ "
        n = len(features)

        for i in range(0, n):
            formula += features[i]
            if not i == (n - 1):
                formula += " + "

        self.formula = formula

    def selectBestFeatures(self, min_include, exclude_intercept=True):

        if self.output.loc[0]['variable'] == "NA":
            self.best_vars = self.original_features
            return (self.best_vars)

        best_vars = list(self.output.loc[self.output['probability'] > min_include]['variable'])

        if exclude_intercept:
            best_vars = list(compress(best_vars, list(map(lambda x: x != '(Intercept)', best_vars))))

        if len(best_vars) < 1:
            print("No features found with good predictive probability")
            best_vars = self.original_features

        self.best_vars = best_vars

        return (self.best_vars)
