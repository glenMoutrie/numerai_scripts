#!/usr/bin/python

from joblib import load
import os
import sys

import pandas as pd

from numerai_analyser import NumeraiConfig
from numerai_analyser import NumeraiDataManager
from numerai_analyser.model_cv import ModelTester


model_repos = os.listdir('model_repo')
model_repos.sort()

run_args = sys.argv
 
if len(run_args) > 1:
	test_run = run_args[1] == 'test_run'

else:
	test_run = True

if not test_run:
	 model_repos = [i for i in model_repos if 'test_run' not in i]

if len(model_repos) < 1:
	raise Error('No viable model repos')

trained_models = load('model_repo/' + model_repos[-1])

print('Running on ' + 'model_repo/' + model_repos[-1])

config_param = dict(**trained_models.test_param, **trained_models.run_param)

config = NumeraiConfig(**config_param)



dm = NumeraiDataManager(config)

train, test = dm.getData(polynomial = config.run_param['polynomial'],
	reduce_features = config.run_param['reduced_features'],
	estimate_clusters = config.run_param['estimate_clusters'],
	use_features = trained_models.features)

trained_models.convert_keras_from_json()

pred_models = [('gbot', 'ensemble'), ('gbot_v2', 'xgbreg_costly'),('gbot_v3', 'DNN_full'),('gbot_v4', 'PLSReg')]

email_body = 'Competition #{}\n'.format(dm.round_num)
email_body += 'Now getting ready to submit the following models:\n'

email_html = pd.DataFrame(pred_models).rename(columns = {0:'model_id', 1:'internal_model_ref'}).to_html()

email_title = "Running Numerai Predict and Submit"
email_title += " (Test Run)" if test_run else ""

config.send_email(body=email_body,html=email_html, attachment=None, header=email_title)

pred_models = [('gbot', 'ensemble'), ('gbot_v2', 'xgbreg_costly'),('gbot_v3', 'DNN_full'),('gbot_v4', 'PLSReg')]
results = {i: ModelTester.getPrediction(config, train, test,  j, trained_models.ensemble_weights, trained_models.trained_models) for i, j in pred_models}
# results = {'gbot': ModelTester.getPrediction(config, train, test,  'ensemble', trained_models.ensemble_weights, trained_models.trained_models),
#            'gbot_v2': ModelTester.getPrediction(config, train, test, 'xgbreg_costly',  trained_models.ensemble_weights, trained_models.trained_models)}

if not test_run:
	dm.submit_results(results, dm.getCompetitions()[0], test)

config.logger.info("Complete.")