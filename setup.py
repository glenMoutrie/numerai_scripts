from setuptools import setup

requirements = [
	'sklearn',
	'pandas',
	'numpy<=1.19.2',
	'xgboost',
	'numerapi',
	'dask',
	'tensorflow',
	'joblib',
	'scipy',
	'joblib',
	'dask[distributed]',
	'psutil',
	'distributed',
	'umap-learn',
	'apricot-select'
]


setup(
	name='numerai_analyser',
	version='0.2',
	description='Automatic predictions for numerai',
	author='Glen Moutrie',
	install_requires=requirements
)

