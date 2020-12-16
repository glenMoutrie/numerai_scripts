from numerai_analyser.analyser import predictNumerai
from numerai_analyser.test_type import TestType as tt

import pytest


class TestEndToEnd:

    def test_synthetic(self):

        predictNumerai(True, tt.SYNTHETIC_DATA)

    @pytest.mark.slow
    def test_subset(self):

        predictNumerai(True, tt.SUBSET_DATA, test_size = 40)
