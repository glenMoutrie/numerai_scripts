# numerai_analyser

This package automates the weekly competitions on [numer.ai](https://numer.ai/).

## Getting Started

```
python run.py
```

## Summary

The package uses [numerapi](https://github.com/uuazed/numerapi) to download the latest competition data, then tests a range of applicable models and features before automatically uploading the predictions from the best model.

## Authors
* **Glen Moutrie**


## Plan:
1. create a data loader that creates a NumeraiData class -- DONE
2. Define a NumeraiData Class with features, x data and y data -- DONE
3. finish the model tester that returns the model results, estimating in parallel using multiprocessing
4. automate with numerapi and other such tools -- DONE
5. Better predictive models, look at alternaitves, betters model specifications
6. Automatic feature selection --DONE
7. Feature engineering (look at clustering etc)
	1. Clustering -- DONE
 	2. Add principal components
 	3. Add predictions from other models
8. Fix the cross terms issue by removing white space in names
9. Potentially try a deep learning approach...
10. Try an ensemble approach accross different epochs
11. Improve unit tests and logging