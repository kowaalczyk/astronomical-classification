import gc
import os
import time
from shutil import copyfile
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd

from plasticc.dataset import Dataset, save_batch, build_dataset_structure


N_JOBS = 16


simple_aggregations = {
    'mjd': ['min', 'max', 'mean', 'count'],  # this is not only time but also a relevant feature, as pointed out on the forum
    'flux': ['min', 'max', 'mean', 'median', 'std', 'skew'],  # all relevant pandas aggregations except count (which is same for all columns and calculated for mjd)
    'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],  # keep these same as for flux - might be useful for future transformations
    'detected': ['mean'],  # this is binary so knowing mean and count translates to knowing how many actual samples were marked as detected
    'flux_ratio_sq': ['min', 'max', 'sum', 'skew'],
    'flux_by_flux_ratio_sq': ['min', 'max', 'sum', 'skew'],
}


def from_base(base_dataset: Dataset, out_dataset_path: str, process_test=True) -> Dataset:
    """
    For each object in metadata, attach min, max and mean values
    for each of the 6 passbands.
    Because we extract features from test time series, the set becomes smaller
    and we squash entire test set back into 1 CSV.
    """
    gc.enable()
    out_dataset = build_dataset_structure(out_dataset_path, with_meta=False)
    train_extraction_time = _process_train_set_count_time(base_dataset, out_dataset)
    gc.collect()
    if process_test:
        print('Processing test CSVs in parallel, there will be no progressbar available.')
        print(f'Processing training set took {train_extraction_time} seconds,')
        print(f'processing full test set should take about {350*(train_extraction_time)/(N_JOBS * 60)} minutes.')
        test_features_df = _process_test_features(base_dataset, out_dataset)
        gc.collect()
        test_out_df = test_features_df.join(base_dataset.test_meta)
        test_out_df.drop(columns=[col for col in set(test_out_df.columns) if col.endswith('_meta')], inplace=True)
        save_batch(test_out_df, output_dir=out_dataset.test_path)
    return out_dataset


def _process_train_set_count_time(base_dataset: Dataset, out_dataset: Dataset) -> float:
    """ 
    Generates features on the train part of the base_dataset, saves it to out_dataset.
    Returns execution time in seconds.
    """
    start_time = time.time()
    train_features_df = _extract_features(
        base_dataset.train_raw,
    )
    train_out_df = train_features_df.join(base_dataset.train_meta)
    train_out_df.drop(columns=[col for col in set(train_out_df.columns) if col.endswith('_meta')], inplace=True)
    train_out_df.to_csv(out_dataset.train_path, index=False)
    end_time = time.time()
    return end_time - start_time


def _process_test_features(base_dataset: Dataset, out_dataset: Dataset) -> pd.DataFrame:
    """Processes test set in parallel and concatenates results into one DataFrame."""
    with Pool(N_JOBS) as pool:
            dfs = pool.map(_read_extract_features, base_dataset.test_paths)
    print('Series processed, joining dataframes...')
    return pd.concat(dfs)


def _read_extract_features(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return _extract_features(df)


def _extract_features(series_df: pd.DataFrame) -> pd.DataFrame:
    series_df = _with_series_features(series_df)
    flat_aggr_df = _compute_aggregate_features(series_df)
    flat_aggr_df = _with_differential_features(flat_aggr_df)
    # TODO: Aggregating min, max, etc. across various passbands might be a good idea, implement it.
    return flat_aggr_df


def _with_series_features(series_df: pd.DataFrame) -> pd.DataFrame:
    series_df['flux_ratio_sq'] = np.power(series_df['flux'] / series_df['flux_err'], 2.0)
    series_df['flux_by_flux_ratio_sq'] = series_df['flux'] * series_df['flux_ratio_sq']
    return series_df


def _compute_aggregate_features(series_df: pd.DataFrame) -> pd.DataFrame:
    aggr_df = series_df.groupby(['passband', 'object_id']).agg(simple_aggregations)
    flattened_dfs = [_flatten_columns(aggr_df.xs(passband), f"passband_{passband}_") for passband in range(1,6)]
    flattened_df = pd.concat(flattened_dfs, axis=1)
    return flattened_df


def _flatten_columns(df: pd.DataFrame, col_prefix: str) -> pd.DataFrame:
    df.columns = [col_prefix + '_'.join(col).strip() for col in df.columns.values]
    return df


def _with_differential_features(flat_aggr_df: pd.DataFrame) -> pd.DataFrame:
    # flux-related features
    for passband in range(1,6):
        for differential_colname in ['flux', 'flux_err']:
            colname_base = f'passband_{passband}_{differential_colname}'
            flat_aggr_df[f'{colname_base}_diff'] = flat_aggr_df[f'{colname_base}_max'] - flat_aggr_df[f'{colname_base}_min']
            flat_aggr_df[f'{colname_base}_diff2'] = flat_aggr_df[f'{colname_base}_diff'] / flat_aggr_df[f'{colname_base}_mean']
        flat_aggr_df[f'passband_{passband}_flux_w_mean'] = flat_aggr_df[f'passband_{passband}_flux_by_flux_ratio_sq_sum'] / flat_aggr_df[f'passband_{passband}_flux_ratio_sq_sum']
        flat_aggr_df[f'passband_{passband}_flux_dif3'] = (flat_aggr_df[f'passband_{passband}_flux_max'] - flat_aggr_df[f'passband_{passband}_flux_min']) \
                                                          / flat_aggr_df[f'passband_{passband}_flux_w_mean']
        # other features
        flat_aggr_df[f'passband_{passband}_detected_count'] = flat_aggr_df[f'passband_{passband}_detected_mean'] * flat_aggr_df[f'passband_{passband}_mjd_count']
        flat_aggr_df[f'passband_{passband}_detected_mjd_diff'] = flat_aggr_df[f'passband_{passband}_mjd_max'] - flat_aggr_df[f'passband_{passband}_mjd_mean']
    return flat_aggr_df
