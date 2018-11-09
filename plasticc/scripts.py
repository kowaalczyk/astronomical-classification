from argparse import ArgumentParser
import shutil
import os

import click
import pandas as pd

from plasticc.dataset import batch_data as batch_data_func
from plasticc.dataset import Dataset
from plasticc import simple


@click.command()
@click.option('--raw-data-path', type=str, 
        help='Path to directory containing raw data', required=True)
@click.option('--base-dataset-path', type=str, 
        help='Output directory path, will be created if necessary.', required=True)
@click.option('--use-sample/--use-full-test', default=True, 
        help='Add --use-full-test flag to perform operation on entire test set')
@click.option('--test-batch-size', default=150000, 
        help='Number of rows in all of the test set batches')
def create_base_dataset(raw_data_path, base_dataset_path, use_sample, test_batch_size):
    out_dataset = Dataset.with_structure(
        base_dataset_path,
        has_meta=True
    )
    print('Copying CSVs...')
    shutil.copyfile(
        os.path.join(raw_data_path, 'training_set.csv'),
        out_dataset.train_path
    )
    shutil.copyfile(
        os.path.join(raw_data_path, 'training_set_metadata.csv'),
        out_dataset.meta_path('train.csv')
    )
    shutil.copyfile(
        os.path.join(raw_data_path, 'test_set_metadata.csv'),
        out_dataset.meta_path('test.csv')
    )
    if use_sample:
        test_csv_name = 'test_set_sample.csv'
        test_csv_lines = 1000001  # for correct progressbar
    else:
        test_csv_name = 'test_set.csv'
        test_csv_lines = 453653105  # for correct progressbar
    print(f"Using {test_csv_name} as a test csv")
    print("Splitting test data into batches...")
    reader = pd.read_csv(
        os.path.join(raw_data_path, test_csv_name), 
        chunksize=test_batch_size
    )
    batch_data_func(
        reader, 
        output_dir=out_dataset.test_path, 
        lines=test_csv_lines
    )
    print('Done.')


@click.command()
@click.option('--base-dataset-path', type=str, 
        help='Path to dataset that will be modified', required=True)
@click.option('--out-dataset-path', type=str, 
        help='Output directory path, will be created if necessary', required=True)
@click.option('--process-test/--ignore-test', default=True, 
        help='Adding flag --ignore-test will create an incomplete dataset without test data')
def featurize_simple(base_dataset_path, out_dataset_path, process_test):
    base_dataset = Dataset(base_dataset_path)
    if not base_dataset.has_meta():
        print("Base dataset must have metadata.")
        exit(1)
    print("Generating features...")
    simple.from_base(base_dataset, out_dataset_path, process_test=process_test)
    print("Done.")
