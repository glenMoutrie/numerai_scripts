from setuptools import setup

setup(
		name='numerai_analyser',
		version='0.1',
		description='Automatic predictions for numerai',
		author='Glen Moutrie',
		install_requires=[
		'sklearn',
		'pandas',
		'numpy',
		'xgboost',
		'numerapi',
		'dask']
		)

