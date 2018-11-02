from setuptools import setup, find_packages

setup(
    name='plasticc',
    version='0.1',
    packages=['plasticc'],
    entry_points={
        'console_scripts': [
            'plasticc-dataset-base = plasticc.scripts:create_base_dataset',
            'plasticc-dataset-simple = plasticc.scripts:featurize_simple'
        ]
    }
)