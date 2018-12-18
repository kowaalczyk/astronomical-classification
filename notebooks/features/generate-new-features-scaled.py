""" Helper script for generating features so that we can examine terminal output after reconnecting. """

from plasticc.featurize import *

import numpy as np
import pandas as pd
from tqdm import tqdm

import multiprocessing as mp
import os
import sys

import gc
gc.enable()


n_jobs=2
meta_path='../../data/raw/test_set_metadata.csv'
test_dir_path='../../data/raw-test-splits/'
output_path='../../data/features-final/test-3band-series-models-without-kernel-features-{}.csv'  # will be formatted


fcp = {
    'flux': {
        'longest_strike_above_mean': None,
        'longest_strike_below_mean': None,
        'mean_change': None,
        'mean_abs_change': None,
        'length': None,
    },

    'flux_by_flux_ratio_sq': {
        'longest_strike_above_mean': None,
        'longest_strike_below_mean': None,       
    },

    'flux_passband': {
        'fft_coefficient': [
                {'coeff': 0, 'attr': 'abs'}, 
                {'coeff': 1, 'attr': 'abs'}
            ],
        'maximum': None, 
        'minimum': None,
        'mean_change': None,
        'mean_abs_change': None,
        'kurtosis' : None, 
        'skewness' : None,
    },

    'mjd': {
        'maximum': None, 
        'minimum': None,
        'mean_change': None,
        'mean_abs_change': None,
    },
}


def featurize_chunk(raw_chunk_csv_path: str) -> pd.DataFrame:
    """ Featurizes a chunk of the dataset and returns featurized DataFrame """
    df = pd.read_csv(raw_chunk_csv_path)
    df_meta = pd.read_csv(meta_path)
    
    preprocessed_df = preprocess_series(df)
    redshift_series = pd.Series(df_meta['hostgal_photoz'].fillna(0).values, index=df_meta['object_id'])
    preprocessed_df = calculate_fixed_passband_and_scaled_flux(preprocessed_df, redshift_series)
    del df  # preprocess series copies the dataframe so we can safely delete it
    gc.collect()
    
    series_features_df = calculate_series_features(preprocessed_df, passband_colname='fixed_passband').add_suffix('_fpb')
    
    agg_df_ts_flux_passband_fpb = extract_features(
        preprocessed_df,
        column_id='object_id',
        column_sort='mjd',
        column_kind='fixed_passband',
        column_value='flux',
        default_fc_parameters=fcp['flux_passband'], 
        n_jobs=n_jobs
    ).add_suffix('_preprocessed')
    agg_df_ts_flux_passband_fpb['object_id'] = agg_df_ts_flux_passband_fpb.index
    result = agg_df_ts_flux_passband_fpb.join(series_features_df, on='object_id')
    result.fillna(0, inplace=True)
    return result


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
