import os
from shutil import copyfile
from itertools import product

import pandas as pd

from plasticc.dataset import Dataset, save_batch


def from_base(base_dataset: Dataset, out_dataset_path: str, process_test=True) -> Dataset:
    """
    For each object in metadata, attach min, max and mean values 
    for each of the 6 passbands.
    Because we extract features from test time series, the set becomes smaller
    and we squash entire set into 1 DataFrame.
    """
    out_dataset = Dataset.with_structure(out_dataset_path, has_meta=False)
    train_out_df = _extract_features(
        base_dataset.train_df, 
        base_dataset.train_meta_df
    )
    train_out_df.to_csv(out_dataset.train_path, index=False)
    if process_test:
        test_meta_df = base_dataset.test_meta_df
        dfs = []  # entire test set will be concatenated into 1 dataframe
        for test_series_df in base_dataset.iter_test_dfs():
            dfs.append(_extract_features(
                test_series_df,
                test_meta_df
            ))
        test_out_df = pd.concat(dfs)
        save_batch(test_out_df, output_dir=out_dataset.test_path)
    return out_dataset


def _extract_features(series_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    gbo = series_df.groupby(['passband', 'object_id'])
    aggr_dfs = {
        'min': gbo.mean(),
        'max': gbo.max(),
        'mean': gbo.mean()
    }
    result_column_names = aggr_dfs['min'].columns
    passbands = range(0,6)
    train_features_df = pd.DataFrame(index=series_df['object_id'].unique())
    for passband, aggr_type, colname in product(passbands, aggr_dfs.keys(), result_column_names):
        out_colname = f'passband_{passband}_{aggr_type}_{colname}'
        train_features_df[out_colname] = aggr_dfs[aggr_type][colname].loc[passband]
    train_out_df = train_features_df.join(meta_df, rsuffix='_meta')
    return train_out_df
