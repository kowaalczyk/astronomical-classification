# astronomical-classification
Working on submission for PLAsTiCC Astronomical Classification

## Directory structure  
```bash
.
├── README.md
.
├── data
│   ├── raw
│   │   ├── test_set_metadata.csv
│   │   ├── test_set_sample.csv
│   │   ├── training_set.csv
│   │   └── training_set_metadata.csv
│   └── sets
│       └── sample  # example of a dataset - all datasets should keep same structure
│           ├── test
│           │   ├── test-batch-000000000013-000000004507.csv
│           │   ├── test-batch-000000004508-000000009115.csv
│           │   ├── test-batch-000000009124-000000014058.csv
│           │   ├── test-batch-000000014061-000000018862.csv
│           │   ├── test-batch-000000018864-000000024106.csv
│           │   ├── test-batch-000000024116-000000028875.csv
│           │   ├── test-batch-000000028877-000000032293.csv
│           │   └── test-batch-000000032300-000000032300.csv
│           └── train.csv  # unlike CSVs in raw data, datasets keep all data joined (metadata with time series if necessary)
├── download-datasets.sh  # script downloading raw data from kaggle
├── notebooks  # visualisation & simple experiments are kept in notebooks here
│   ├── Batch data.ipynb
│   └── Exploratory.ipynb
├── plasticc  # python package containing all scripts and modules that can be imported into notebooks
│   ├── __init__.py
│   ├── dataset.py
│   └── scripts.py
├── requirements-dev.txt
├── requirements.txt
└── setup.py  # build information for plasticc package
```

## Dev setup

1. Create and activate python virtual environment (for conda, use `conda create -n plasticc python=3.6 ipykernel` and then `source activate plasticc` (linux) or `activate plasticc` (windows))  
2. Install requirements: `pip install -r requirements.txt`  
and dev requirements `pip install -r requirements-dev.txt`  
3. Download datasets (~7GB, this may take very long): `bash download-datasets.sh`  
4. Install `plasticc` package: `pip install -e .` or `python setup.py develop` 
(neither version works on Windows, unfortunately). 
The package will be auto-reloaded so you only need to do this once.  

After that, to recreate dataset structure described above (using test_set_sample.csv for local development):  
```bash
mkdir data/sets/sample  
# this might take some time:  
plasticc-batch-data --input-csv data/raw/test_set_sample.csv --output-dir data/sets/sample/test/ --input-rows 1000001  
cp data/raw/training_set.csv data/sets/sample/train.csv  
```
Note that this serves only demonstrational purposes - it does not make much sense 
to create a dataset without any metadata (just the time series).

## Rules  

- All important code should be in the `plasticc` package  
- Use PEP-8 standard for all code and keep lines 80-90 chars long at maximum  
- Do not push changes to `plasticc` package directly into master - 
create a branch, make a pull request and make someone review it before merge  
- All experiments should consist of calls to scripts from `plasticc` only. 
This will allow us to reproduce the experiment by executing same commands with same parameters. 
Before starting any serious research, we will work out specific tools for experiment tracking 
(most likely DVC or Makefiles).  
