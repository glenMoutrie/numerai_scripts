from numerai_analyser.analyser import predictNumerai
from numerai_analyser.test_type import TestType

# Route map for model_improvements branch
# 1) Improve logging so that you can assess improvement of implementation
#       a) More measurements of predictive accuarcy - DONE
#       b) Bench mark model run times - DONE
#       c) Break down of era's with good/bad predictions - DONE
#       d) Potentially consider implementing postgres db...
#
# 2) Model selection
#       a) Create a better voting system for model selection, consider going for an ensemble approach - DONE
#       b) This needs to rely somewhat on the architecture used for logging - DONE
#       c) Better hyperparamter selection - DONE(ish)
#       d) Maybe... maybe implement dnn... - DONE
#       e) incorporate boom spike slab without polynomial - DONE
#       f) use numerai_score and sharpe ratio for model selection
#               i) create numerai score in sklearn fashion
#               ii) Incorporate into CV for multiple components
#       g) cross validate for xgboost parameters and others
#       h) improve model weights and selection for ensemble
#       i) better feature selection, try xgboost feature importance
#
# 3) Performance
#       a) Better parallelisation for model estimation - DONE
#       b) multiprocessing/dask for different cuts of the data
#       c) Better unit testing - IN PROGRESS
#       d) Rethink data sets - IN PROGRESS
#              i) easier creation in data manager - DONE
#              ii) don't use era's as categories - DONE
#              iii) make getX and getY cleaner with shared logic between test and train
#              iv) create a better transformation step (umap, pca, polynomial options)
#              v) apply over era function

if __name__ == "__main__":
	# predictNumerai(True, TestType.SYNTHETIC_DATA, 2000, splits = 2, email_updates = False)
	# predictNumerai(True, TestType.SUBSET_DATA, 2000, splits = 3, email_updates = False)
	predictNumerai(False, splits = 1)

