from typing import Tuple

import pandas as pd
import tsfresh
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters

from plasticc.dataset import Dataset, build_dataset_structure, save_batch


def from_base(base_dataset: Dataset, out_dataset_path: str, features_dict: dict=None, process_test=True) -> Dataset:
    """ Features dict contains parameters passed to tsfresh, if None is passed all possible features are extracted. """
    out_dataset = build_dataset_structure(out_dataset_path, with_meta=False)
    train_out_df, na_colnames = _extract_features(
        base_dataset.train_raw, 
        base_dataset.train_meta,
        features_dict
    )
    # train df will be saved later for it to have same features as test df
    if process_test:
        test_meta_df = base_dataset.test_meta
        dfs = []  # entire test set will be concatenated into 1 dataframe
        for test_series_df in base_dataset.iter_test:
            batch_df, batch_na_colnames = _extract_features(
                test_series_df,
                test_meta_df,
                features_dict
            )
            dfs.append(batch_df)
            na_colnames |= batch_na_colnames
        test_out_df = pd.concat(dfs)
        test_out_df.drop(columns=na_colnames&set(test_out_df.columns), inplace=True)
        save_batch(test_out_df, output_dir=out_dataset.test_path)
    # before saving, remove columns that were na in any of test batches:
    train_out_df.drop(columns=na_colnames&set(train_out_df.columns), inplace=True)
    train_out_df.to_csv(out_dataset.train_path, index=False)


def _extract_features(series_df: pd.DataFrame, meta_df: pd.DataFrame, features_dict: dict) -> Tuple[pd.DataFrame, set]:
    # perform feature extraction
    features = tsfresh.extract_features(
        series_df,
        column_id='object_id',
        column_sort='mjd',
        column_kind='passband',
        column_value='flux',
        default_fc_parameters=features_dict
    )
    na_feature_names = set(col for col in features.columns if features[col].isna().any())
    features.dropna(axis=1, inplace=True)  # remove unnecessary features asap
    # rename id and index of the features to 'object_id':
    features['object_id'] = features.index
    features.index = features['object_id']
    out_df = features.join(meta_df, rsuffix='_meta')
    return out_df, na_feature_names  # na_features might have to be removed from other dataframes in the set
