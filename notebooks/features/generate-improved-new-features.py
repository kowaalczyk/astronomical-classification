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


meta_path='../../data/raw/test_set_metadata.csv'
test_dir_path='../../data/raw-test-splits/'
output_path='../../data/features-final/impoved-3band-series-models-without-kernel-features-{}.csv'  # will be formatted
n_jobs=2


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
    df = pd.read_csv(raw_chunk_csv_path)
    df_meta = pd.read_csv('../../data/raw/test_set_metadata.csv')
    
    
    # start by scaling time and fixing passband alignment
    print("Preprocessing time...")
    preprocessed_df = preprocess_series(df)
    gc.collect()
    print("Fixing passband alignment...")
    redshift_series = pd.Series(df_meta['hostgal_photoz'].fillna(0).values, index=df_meta['object_id'])
    preprocessed_df = calculate_fixed_passband_and_scaled_flux(preprocessed_df, redshift_series)
    
    # new, custom series features
#     print("Generating custom features...")
#     series_features_df = calculate_series_features(preprocessed_df)
#     print("Generating custom features for fixed passbands...")
#     series_features_df_fpb = calculate_series_features(preprocessed_df, passband_colname='fixed_passband').add_suffix('_fpb')
#     print("Custom features generated.")
#     total_nans = series_features_df.isna().any().sum()
#     if total_nans > 0:
#         print(f"WARNING: number of NaNs: {total_nans}")
#         series_features_df.fillna(0, inplace=True)

    # features from the kernel
    
    df = process_flux(df)

#     agg_df = df.groupby('object_id').agg(aggs)
#     agg_df.columns = ['{}_{}'.format(k, agg)
#                       for k in aggs.keys() for agg in aggs[k]]
#     agg_df = process_flux_agg(agg_df)  # new feature to play with tsfresh

    # Add more features with tsfresh:
    agg_df_ts_flux_passband = extract_features(
        df,
        column_id='object_id',
        column_sort='mjd',
        column_kind='passband',
        column_value='flux',
        default_fc_parameters=fcp['flux_passband'], 
        n_jobs=n_jobs
    )
    agg_df_ts_flux_passband_fpb = extract_features(
        preprocessed_df,
        column_id='object_id',
        column_sort='mjd',
        column_kind='fixed_passband',
        column_value='flux',
        default_fc_parameters=fcp['flux_passband'], 
        n_jobs=n_jobs
    ).add_suffix('_preprocessed')
    agg_df_ts_flux = extract_features(
        df,
        column_id='object_id',
        column_value='flux',
        default_fc_parameters=fcp['flux'],
        n_jobs=n_jobs
    )
#     agg_df_ts_flux_by_flux_ratio_sq = extract_features(
#         df,
#         column_id='object_id',
#         column_value='flux_by_flux_ratio_sq',
#         default_fc_parameters=fcp['flux_by_flux_ratio_sq'],
#         n_jobs=n_jobs
#     )
    # Add smart feature that is suggested here
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
    # dt[detected==1, mjd_diff:=max(mjd)-min(mjd), by=object_id]
#     df_det = df[df['detected'] == 1].copy()
#     agg_df_mjd = extract_features(
#         df_det,
#         column_id='object_id',
#         column_value='mjd',
#         default_fc_parameters=fcp['mjd'],
#         n_jobs=n_jobs
#     )
#     agg_df_mjd['mjd_diff_det'] = agg_df_mjd['mjd__maximum'].values - agg_df_mjd['mjd__minimum'].values
#     del agg_df_mjd['mjd__maximum'], agg_df_mjd['mjd__minimum']

    agg_df_ts_flux_passband.index.rename('object_id', inplace=True)
    agg_df_ts_flux.index.rename('object_id', inplace=True)
    agg_df_ts_flux_passband_fpb.index.rename('object_id', inplace=True)
#     agg_df_ts_flux_by_flux_ratio_sq.index.rename('object_id', inplace=True)
#     agg_df_mjd.index.rename('object_id', inplace=True)
    result = pd.concat([
#         agg_df, 
        agg_df_ts_flux_passband,
        agg_df_ts_flux,
        agg_df_ts_flux_passband_fpb,
#         agg_df_ts_flux_by_flux_ratio_sq,
#         agg_df_mjd
    ],
        axis=1
    ).reset_index()
#     result = agg_df_ts.merge(right=df_meta, how='left', on='object_id')
    result.fillna(0, inplace=True)
    
#     result = result.join(series_features_df, on='object_id')  # newly added series features
#     result = result.join(series_features_df_fpb, on='object_id')  # newly added series features
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
