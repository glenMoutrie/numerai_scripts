from numerai_analyser import NumeraiConfig
from numerai_analyser import NumeraiDataManager

import tempfile
import shutil
import random

import pandas as pd

class TestDataManagerConnectivity():

    config = NumeraiConfig(test_run=False, save_log_file=False, key_loc=None)
    dm = NumeraiDataManager(config)

    def test_connected(self):
        assert self.dm.connected

    def test_correct_round_num(self):
        assert self.dm.round_num == self.dm.api_conn.get_current_round()

    def test_download_latest(self):

        temp_dir = tempfile.mkdtemp()
        try:
            self.dm._get_new_dataset(temp_dir)
        finally:
            shutil.rmtree(temp_dir)

    def test_use_previously_downloaded(self):

        self.dm.downloadLatest()
        comp = self.dm.getCompetitions()[0]

        train, test = self.dm.getData(competition_type= comp, polynomial=False, reduce_features=False)

    def test_upload_dataset(self):

        comp = self.dm.getCompetitions()[0]
        train, test = self.dm.getData(competition_type=comp, polynomial=False, reduce_features=False)

        results_col = 'probability_' + comp

        results = [random.uniform(0,1) for i in test.getID()]

        results_df = pd.DataFrame(data={results_col: results})
        results_df = pd.DataFrame(test.getID()).join(results_df)

        try:
            self.dm.uploadResults(results_df, comp)
        except ValueError as error:
            print(error)




