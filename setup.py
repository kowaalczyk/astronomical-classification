from setuptools import setup, find_packages

setup(
    name='plasticc',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'plasticc-dataset-base = plasticc.scripts:create_base_dataset',
            'plasticc-dataset-simple = plasticc.scripts:featurize_simple',
            'plasticc-dataset-tsfresh = plasticc.scripts:featurize_tsfresh',
            'plasticc-train = plasticc.scripts:train',
            'plasticc-predict = plasticc.scripts:predict'
        ]
    }
)
