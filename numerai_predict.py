from joblib import load
import os

from numerai_analyser import NumeraiConfig
from numerai_analyser import NumeraiDataManager
from numerai_analyser.model_cv import ModelTester


model_repos = os.listdir('model_repo')
model_repos.sort()

test_run = True

if not test_run:
	 model_repos = [i for i in model_repos if 'test_run' not in i]

if len(model_repos) < 1:
	raise Error('No viable model repos')

trained_models = load('model_repo/' + model_repos[-1])


config_param = dict(**trained_models.test_param, **trained_models.run_param)

config = NumeraiConfig(**config_param)

dm = NumeraiDataManager(config)

train, test = dm.getData(polynomial = config.run_param['polynomial'],
	reduce_features = config.run_param['reduced_features'],
	estimate_clusters = config.run_param['estimate_clusters'],
	use_features = trained_models.features)

trained_models.convert_keras_from_json()

results = {'gbot': ModelTester.getPrediction(config, train, test,  'ensemble', trained_models.ensemble_weights, trained_models.trained_models),
           'gbot_v2': ModelTester.getPrediction(config, train, test, 'xgbreg_costly',  trained_models.ensemble_weights, trained_models.trained_models)}

if not test_run:
	dm.submit_results(results, dm.getCompetitions()[0], test)

config.logger.info("Complete.")