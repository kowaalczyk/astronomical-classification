import os
from shutil import copyfile
from itertools import product

import numpy as np
import pandas as pd

from plasticc.dataset import Dataset, save_batch, build_dataset_structure


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
    out_dataset = build_dataset_structure(out_dataset_path, with_meta=False)
    train_out_df = _extract_features(
        base_dataset.train_raw,
        base_dataset.train_meta
    )
    train_out_df.to_csv(out_dataset.train_path, index=False)
    if process_test:
        test_meta_df = base_dataset.test_meta
        dfs = []  # entire test set will be concatenated into 1 dataframe
        for test_series_df in base_dataset.iter_test:
            dfs.append(_extract_features(
                test_series_df,
                test_meta_df
            ))
        test_out_df = pd.concat(dfs)
        save_batch(test_out_df, output_dir=out_dataset.test_path)
    return out_dataset


def _extract_features(series_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    series_df = _with_series_features(series_df)
    flat_aggr_df = _compute_aggregate_features(series_df)
    flat_aggr_df = _with_differential_features(flat_aggr_df)
    # TODO: Aggregating min, max, etc. across various passbands might be a good idea, implement it.
    joined_df = flat_aggr_df.join(meta_df, rsuffix='_meta')
    return joined_df.drop(columns=[col for col in set(joined_df.columns) if col.endswith('_meta')])


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
