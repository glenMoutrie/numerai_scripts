from setuptools import setup

requirements = [
	'sklearn',
	'pandas',
	'numpy',
	'xgboost',
	'numerapi',
	'dask',
	'rpy2',
	'tensorflow',
	'joblib',
	'scipy',
	'joblib'
]


setup(
	name='numerai_analyser',
	version='0.1',
	description='Automatic predictions for numerai',
	author='Glen Moutrie',
	install_requires=requirements
)

