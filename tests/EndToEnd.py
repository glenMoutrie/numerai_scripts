import unittest
from numerai_analyser.analyser import predictNumerai
from numerai_analyser.test_type import TestType

class EndToEnd(unittest.TestCase):

	def test(self):
		predictNumerai(True, TestType.SYNTHETIC_DATA)
		predictNumerai(True, TestType.SUBSET_DATA)