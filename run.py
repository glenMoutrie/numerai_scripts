from numerai_analyser.analyser import predictNumerai
from numerai_analyser.test_type import TestType

if __name__ == "__main__":
	# predictNumerai(True, TestType.SYNTHETIC_DATA, 50000)
	predictNumerai(False)
