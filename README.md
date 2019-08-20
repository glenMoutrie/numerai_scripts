# Numer.ai analyser

This is a small repo to store the scripts for Numer.ai competitions. 

I didn't intend to build this out too heavily, but as I have added features it made sense to build them out in a class structure, this is now becoming large enough to merit a package.

There are a range of procedures in here, but in summary running analyser.py will run the whole process. It will look for an api_key file in the same directory and download the latest data set.

Once all of the data is downloaded and cleaned the features are automatically selected and a range of models are tested. For feature selection boomspikeslab in R is used (there isn't an obvious python equivalent), this is called from feature_selection.py.

Finally the results are pushed out to numerai upon once the best model has been selected.
