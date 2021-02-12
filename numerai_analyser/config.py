import os
import configparser
from datetime import datetime
import logging
from .test_type import TestType
from .email_manager import EmailManager
from pathlib import Path
import psutil

class NumeraiConfig():

    def __init__(self, test_run = True, test_type = TestType.SYNTHETIC_DATA, test_size = 100, save_log_file = False,
                 key_loc = None, email_updates = True, **kwargs):

        self.start_time = datetime.now()

        self.time_file_safe = self.start_time.strftime("%Y_%m_%d_%H%M%S")

        self.start_time = self.start_time.strftime("%Y-%m-%d_%H:%M:%S")

        self.test_run = test_run

        self.save_log_file = save_log_file

        self.key_loc = key_loc

        self.email_updates = email_updates

        self.user = None

        self.key = None

        self.n_cores = psutil.cpu_count(False)

        additional_params = kwargs

        self.run_param = self.setupRunParams(test_run, **additional_params)

        os.environ["OMP_NUM_THREADS"] = str(self.n_cores)

        if test_run:

            self.test_type = test_type
            self.test_size = test_size

        else:

            self.test_type = self.test_size = None

        self.test_param = {'test_run': test_run,
                           'test_type': test_type,
                           'test_size': test_size}

        self.setup()

    def setupRunParams(self, test_run, **kwargs):

        default_params = {
            'splits' : 2 if test_run else 10,
            'n_est' : 200 if test_run else 2000,# numerai recomendation is 2000 but takes ~4hrs+ per fit
            'param_splits' : 2 if test_run else 10,
            'split_test_size' : 0.25,
            'polynomial' : True,
            'reduced_features' : True,
            'estimate_clusters' : False
        }

        for i in default_params.keys():
            if i in kwargs.keys():
                default_params[i] = kwargs[i]

        return default_params


    def setupLogger(self):

        self.logger = logging.getLogger('numerai_run')

        log_file = logging.FileHandler(filename = self.log_text_file, mode = 'a')

        log_format = logging.Formatter('%(asctime)s %(name)s %(levelname)s:\t%(message)s')

        log_file.setFormatter(log_format)

        if self.save_log_file:
            self.logger.addHandler(log_file)

    def setup(self):

        self.setupDirectories()

        self.setupLogger()

        if self.key_loc.exists():

            self.readKey(self.key_loc)

        else:
            self.key_loc = None

        if self.test_run:

            out = 'TEST RUN: '

            if self.test_type is TestType.SYNTHETIC_DATA:
                out += 'synthetic data test'
            elif self.test_type is TestType.SUBSET_DATA:
                out += 'subset data test'

            self.logger.info(out)

    def setupDirectories(self):

        if self.key_loc is not None:

            self.key_loc = Path(self.key_loc)

        else:

            self.numerai_home = Path(os.getcwd())

        test_flag = "test_run_" if self.test_run  else ""

        self.key_loc = self.numerai_home / "api_key"

        self.download_loc = setupDir(self.numerai_home / "datasets")

        self.config_loc = setupDir(self.numerai_home / "logs" / "runtime")

        self.metric_loc = setupDir(self.numerai_home / "logs" / "model_performance")

        self.model_repo = setupDir(self.numerai_home / "model_repo")

        self.metric_loc_file = self.metric_loc / ("metric_log_" + test_flag + self.time_file_safe + ".csv")

        self.model_save_file = self.model_repo / ("model_save_" + test_flag + self.time_file_safe + ".joblib")

        self.log_text_file = self.config_loc / (test_flag + self.time_file_safe + ".txt")


    def readKey(self, key_loc):

        conf_parser = configparser.ConfigParser()
        conf_parser.read(key_loc)

        self.user = conf_parser['DEFAULT']['user']
        self.key = conf_parser['DEFAULT']['key']

        self.email_updates = bool(conf_parser['DEFAULT']['email_updates']) & self.email_updates

        if self.email_updates:

            sender_email = conf_parser['DEFAULT']['sender_email']
            email_pass = conf_parser['DEFAULT']['email_pass']
            receiver_email = conf_parser['DEFAULT']['receiver_email']

            self.email_manager = EmailManager(sender_email= sender_email,
                                              password = email_pass,
                                              receiver_email = receiver_email)



    def shutdown(self):
        self.logger.shutdown()

    def send_email(self, body = None, html = None, header = None, attachment = None):

        if self.email_updates:
            self.email_manager.send_email(body, html, header, attachment)
            self.logger.info('Email "{}" sent'.format(header))

        else:
            self.logger.info('Email update toggled to not send')


def setupDir(path):

    if not path.exists():

        path.mkdir()

    return path