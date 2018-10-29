from setuptools import setup, find_packages

setup(
    name='plasticc',
    version='0.1',
    packages=['plasticc'],
    entry_points={
        'console_scripts': [
            'plasticc-batch-data = plasticc.scripts:batch_data'
        ]
    }
)