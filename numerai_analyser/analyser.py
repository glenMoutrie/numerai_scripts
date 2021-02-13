#!numerai/bin/python

from .model_cv import ModelTester
from .config import NumeraiConfig
from .data_manager import NumeraiDataManager
from .test_type import TestType
from .model_factory import ModelFactory
from .trained_model_io import save_numerai_run

import sys


def predictNumerai(test_run = False, test_type = TestType.SYNTHETIC_DATA, test_size = 100, splits = 3, email_updates = True):

    config = NumeraiConfig(test_run, test_type, test_size, save_log_file= True, email_updates = email_updates, splits = splits)

    try:

        dl = NumeraiDataManager(config)

        competitions = dl.getCompetitions()

        config.logger.info('Running on the following competitions: ' + ', '.join(competitions))

        for comp in competitions:

            config.logger.info('Running on comp ' + comp)

            train, test = dl.getData(competition_type = comp,
                                     polynomial = config.run_param['polynomial'],
                                     reduce_features = config.run_param['reduced_features'],
                                     estimate_clusters = config.run_param['estimate_clusters'])

            email_body = """Feature selection completed for round {0} competition {1}. 
Now running paramter cross validation.""".format(dl.round_num, comp)
            email_title = 'Numerai Round {0} update'.format(dl.round_num)

            config.send_email(body = email_body,
                              html = None,
                              attachment = None,
                              header = email_title)

            mf = ModelFactory(config.run_param['n_est'])

            config.logger.info('Estimating costly models')

            mf.estimate_costly_models(train)

            config.logger.info('Performing cross validation')

            mf.cross_validate_model_params(train, config.run_param['param_splits'], n_cores = config.n_cores)

            email_body = """Model parameterization completed for round {0} competition {1}.
Now running model testing over {2} splits.""".format(dl.round_num, comp, config.run_param['splits'])

            config.send_email(body=email_body,
                              html=None,
                              attachment=None,
                              header=email_title)

            tester = ModelTester(mf, train.getEras(unique_eras = True), config,
                                 config.run_param['splits'], config.run_param['split_test_size'])

            tester.testAllSplits(train)

            tester.train_all_models(train)

            save_numerai_run(config, tester.getTrainedOutput(train.features))

            results = {'gbot': tester.getPrediction(config, train, test,  tester.getBestModel(), tester.getEnsembleWeights(), tester.trained_models),
                       'gbot_v2': tester.getPrediction(config, train, test, 'xgbreg_costly',  tester.getEnsembleWeights(), tester.trained_models),
                       'gbot_v3': tester.getPrediction(config, train, test, 'DNN_full',  tester.getEnsembleWeights(), tester.trained_models),
                       'gbot_v4': tester.getPrediction(config, train, test, 'PLSReg',  tester.getEnsembleWeights(), tester.trained_models)}

            if not test_run:

                dl.submit_results(results, comp, test)

            config.logger.info("Complete.")


    # except:
    #     email_body = """The run has reached hit an error for round {0}\n{1}""".format(dl.round_num,
    #                                                                                sys.exc_info()[0])
    #     email_title = 'Numerai Round {0} error'.format(dl.round_num)
    #
    #     config.send_email(body=email_body,
    #                       html=None,
    #                       attachment=None,
    #                       header=email_title)


    finally:

        email_body = """The run has reached an end for round {0}""".format(dl.round_num)
        email_title = 'Numerai Round {0} run finished'.format(dl.round_num)

        config.send_email(body=email_body,
                          html=None,
                          attachment=None,
                          header=email_title)



