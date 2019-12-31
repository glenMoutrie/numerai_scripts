from numerai_analyser.analyser import predictNumerai
from numerai_analyser.test_type import TestType

if __name__ == "__main__":
	predictNumerai(True, TestType.SUBSET_DATA, 300)
	# predictNumerai(False)