import os
from datetime import datetime
import logging
from .test_type import TestType
from pathlib import Path

class NumeraiConfig():

    def __init__(self, test_run, test_type):

        self.start_time = datetime.now()

        self.time_file_safe = self.start_time.strftime("%Y_%m_%d_%H%M%S")

        self.start_time = self.start_time.strftime("%Y-%m-%d_%H:%M:%S")

        self.test_run = test_run

        if test_run:

            self.test_type = test_type

        else:

            self.test_type = None

    def setupLogger(self):

        self.logger = logging.getLogger('numerai_run')

        # sys_out = logging.StreamHandler()
        log_file = logging.FileHandler(filename = self.log_text_file, mode = 'a')

        log_format = logging.Formatter('%(asctime)s %(name)s %(levelname)s:\t%(message)s')

        # sys_out.setFormatter(log_format)
        log_file.setFormatter(log_format)

        # self.logger.addHandler(sys_out)
        self.logger.addHandler(log_file)

    def setup(self):

        self.setupDirectories()

        self.setupLogger()

        os.environ["OMP_NUM_THREADS"] = "8"

        if self.test_run:

            out = 'TEST RUN: '

            if self.test_type is TestType.SYNTHETIC_DATA:
                out += 'synthetic data test'
            elif self.test_type is TestType.SUBSET_DATA:
                out += 'subset data test'

            self.logger.info(out)

    def setupDirectories(self):

        self.numerai_home = Path(os.getcwd())

        self.key_loc = self.numerai_home / "api_key"

        self.download_loc = setupDir(self.numerai_home / "datasets")

        self.config_loc = setupDir(self.numerai_home / "logs" / "runtime")

        self.metric_loc = setupDir(self.numerai_home / "logs" / "model_performance")

        self.metric_loc_file = self.metric_loc / ("metric_log_" + self.time_file_safe + ".csv")

        self.log_text_file = self.config_loc / (self.time_file_safe + ".txt")



    def shutdown():
        self.logger.shutdown()


def setupDir(path):

    if not path.exists():

        path.mkdir()

    return path