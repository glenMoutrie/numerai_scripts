from numerai_analyser.analyser import predictNumerai
from numerai_analyser.test_type import TestType as tt
import pytest

class TestEndToEnd:

	def test_synthetic(self):

		try:
			predictNumerai(True, tt.SYNTHETIC_DATA)
		except Exception as e:
			pytest.fail(e)

	@pytest.mark.slow
	def test_subset(self):

		try:
			predictNumerai(True, tt.SUBSET_DATA, test_size = 40)
		except Exception as e:
			pytest.fail(e)
