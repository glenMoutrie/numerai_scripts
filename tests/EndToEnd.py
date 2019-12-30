import unittest
from numerai_analyser.analyser import predictNumerai

class EndToEnd(unittest.TestCase):

	def test(self):
		predictNumerai(True)