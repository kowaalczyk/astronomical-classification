import shutil
import os
import pickle

import click
import pandas as pd
import numpy as np

from plasticc.dataset import batch_data as batch_data_func
from plasticc.dataset import Dataset, build_dataset_structure
from plasticc.features import simple, tsfresh
# from plasticc import train_xgb
from plasticc import metrics
from plasticc import submission


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
    out_dataset = build_dataset_structure(
        base_dataset_path,
        with_meta=True
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
    base_dataset = Dataset(base_dataset_path, "target")
    if not base_dataset.has_meta():
        print("Base dataset must have metadata.")
        exit(1)
    print("Generating features...")
    simple.from_base(base_dataset, out_dataset_path, process_test=process_test)
    print("Done.")

    
@click.command()
@click.option('--base-dataset-path', type=str, 
        help='Path to dataset that will be modified', required=True)
@click.option('--out-dataset-path', type=str, 
        help='Output directory path, will be created if necessary', required=True)
@click.option('--tsfresh-config-path', type=str, 
        help='Path to tsfresh settings dict, if not provided it will extract all possible features', default=None)
@click.option('--process-test/--ignore-test', default=True, 
        help='Adding flag --ignore-test will create an incomplete dataset without test data')
def featurize_tsfresh(base_dataset_path, out_dataset_path, tsfresh_config_path, process_test):
    base_dataset = Dataset(base_dataset_path, "target")
    if not base_dataset.has_meta():
        print("Base dataset must have metadata.")
        exit(1)
    tsfresh_config = None
    if tsfresh_config_path is None:
        tsfresh_config = {
            'fft_coefficient': [{'coeff': 0, 'attr': 'abs'}, {'coeff': 1, 'attr': 'abs'}],
            'kurtosis' : None,
        }
    else:
        with open(tsfresh_config_path, 'rb') as file:
            tsfresh_config = pickle.load(file)
    print("Generating features...")
    tsfresh.from_base(base_dataset, out_dataset_path, features_dict=tsfresh_config, process_test=process_test)
    print("Done.")


@click.command()
@click.option('--dataset-path', help="Dataset to be trained", required=True)
@click.option('--output-path', help="Model output", required=True)
@click.option('--calc-score', help="Should I calculate score", default=True)
@click.option('--target-col', help="Name of target column", default="target")
@click.option('--cv-scoring', help="Scoring algorithm", default="f1_macro")
@click.option('--cv-splits', help="Number of CV splits", default=5)
@click.option('--cv-test-size', help="Size of CV test part", default=0.2)
@click.option('--xgb-max-depth', help="XGB max depth", default=7)
@click.option('--xgb-lr', help="XGB learning rate", default=0.1)
def train_xgboost(
        dataset_path: str,
        output_path: str,
        calc_score: bool,
        target_col: str,
        cv_scoring: str,
        cv_splits: int,
        cv_test_size: float,
        xgb_max_depth: int,
        xgb_lr: float,
):
    xgb_args = {
        "max_depth": xgb_max_depth,
        "learning_rate": xgb_lr,
        "missing": np.nan,
        "error_score": "raise"
    }
    dataset = Dataset(dataset_path, target_col)

    if calc_score:
        print("Calculating score...")
        cv_args = {"n_splits": cv_splits,
                   "test_size": cv_test_size,
                   "random_state": 2137}
        scores = metrics.xgb_score(dataset, cv_args, xgb_args, cv_scoring)
        print("Scores: ", scores)

    print("Starting training...")
    train_xgb.train(dataset, output_path, xgb_args)
    print("Successfully trained. Dumped results into ", output_path)


@click.command()
@click.option("--model", "model_path", type=str, help="Saved model path", required=True)
@click.option("--input", "dataset_path", type=str, help="Input path", required=True)
@click.option("--output", "output_path", type=str, help="Output path", default="/dev/stdout")
def eval_model(model_path: str, dataset_path: str, output_path: str):
    print("Loading model...")
    model = submission.load_model(model_path)
    print("Loading dataset...")
    data = Dataset(dataset_path)
    print("Artifical contemplation...")
    submission.prepare_submission(output_path, model, data)
    print("Done.")
