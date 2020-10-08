#!numerai/bin/python
import pandas as pd

from .model_cv import ModelTester
from .config import NumeraiConfig
from .data_manager import NumeraiDataManager
from .test_type import TestType
from .model_factory import ModelFactory

import urllib3
import requests
import sys


def predictNumerai(test_run = False, test_type = TestType.SYNTHETIC_DATA, test_size = 100, splits = 3):

    config = NumeraiConfig(test_run, test_type, test_size, save_log_file= True)

    try:

        dl = NumeraiDataManager(config)

        competitions = dl.getCompetitions()

        config.logger.info('Running on the following competitions: ' + ', '.join(competitions))

        for comp in competitions:

            config.logger.info('Running on comp ' + comp)

            train, test = dl.getData(competition_type = comp,  polynomial = False, reduce_features = True)

            email_body = """Feature selection completed for round {0} competition {1}. 
Now running paramter cross validation.""".format(dl.round_num, comp)
            email_title = 'Numerai Round {0} update'.format(dl.round_num)

            config.send_email(body = email_body,
                              html = None,
                              attachment = None,
                              header = email_title)

            if test_run:
                n_est = 200
                cv_splits = 2

            else:

                n_est = 1000 # numerai recomendation is 20000 but takes ~4hrs+ per fit
                cv_splits = 10

            mf = ModelFactory(n_est)

            mf.cross_validate_model_params(train, cv_splits)

            email_body = """Model parameterization completed for round {0} competition {1}.
Now running model testing over {2} splits.""".format(dl.round_num, comp, splits)

            config.send_email(body=email_body,
                              html=None,
                              attachment=None,
                              header=email_title)

            tester = ModelTester(mf, train.getEras(unique_eras = True), config, splits, 0.25)

            tester.testAllSplits(train)

            results = tester.getBestPrediction(train, test)

            results_col = 'probability_' + comp

            results_df = pd.DataFrame(data={results_col: results})
            results_df = pd.DataFrame(test.getID()).join(results_df)

            if not test_run:

                dl.uploadResults(results_df, comp)

                try:
                    dl.getSubmissionStatus()
                except (ValueError,
                        OSError,
                        IOError,
                        urllib3.exceptions.ProtocolError,
                        requests.exceptions.ConnectionError) as error:
                    config.logger.error("Caught error in upload for " + comp)
                    config.logger.error(error)

            config.logger.info("Complete.")

    finally:

        email_body = """The run has reached an end for round {0}""".format(dl.round_num)
        email_title = 'Numerai Round {0} run finished'.format(dl.round_num)

        config.send_email(body=email_body,
                          html=None,
                          attachment=None,
                          header=email_title)



