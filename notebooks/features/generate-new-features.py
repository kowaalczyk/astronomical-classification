""" Helper script for generating features so that we can examine terminal output after reconnecting. """

from plasticc.featurize import preprocess_series, process_group

import numpy as np
import pandas as pd
from tqdm import tqdm

import multiprocessing as mp
import os
import sys

import gc
gc.enable()


meta_path='../../data/raw/test_set_metadata.csv'  # unused right now
test_dir_path='../../data/raw/test-splits/'
output_path='../../data/features/test-3band-series-models-without-kernel-features-{}.csv'  # will be formatted


def featurize_chunk(raw_chunk_csv_path: str) -> pd.DataFrame:
    """ Featurizes a chunk of the dataset and returns featurized DataFrame """
    df = pd.read_csv(raw_chunk_csv_path)
    preprocessed_df = preprocess_series(df)
    del df  # preprocess series copies the dataframe so we can safely delete it
    gc.collect()
    
    gbo = preprocessed_df.groupby('object_id')
    group_features_series = [process_group(g) for g in gbo]
    series_features_df = pd.concat(group_features_series, axis=1)
    return series_features_df.transpose()  # will be joined with other features at the end


if __name__ == '__main__':
    """ Featurizes 1/n_workers of the files in output_path, CSVs to process are selected using this_worker_idx. """
    this_worker_idx = int(sys.argv[1])
    n_workers = int(sys.argv[2])
    print(f"Worker {this_worker_idx} out of {n_workers}")
    
    test_paths = sorted([os.path.join(test_dir_path, filename) for filename in os.listdir(test_dir_path)])
    this_worker_test_paths = [path for idx, path in enumerate(test_paths) if idx % n_workers == this_worker_idx]
    
    print('Performing featurization...', flush=True)
    featurized_dfs = [featurize_chunk(path) for path in tqdm(this_worker_test_paths)]

    print('Concatenating...')
    featuirized_df = pd.concat(featurized_dfs)
    featuirized_df.to_csv(output_path.format(this_worker_idx), index=True)
    print(f'Output saved to {output_path.format(this_worker_idx)}')
    print('Done!')
