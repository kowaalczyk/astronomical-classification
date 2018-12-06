import gc
import time

import numpy as np
import pandas as pd

from plasticc.training import predict_chunk
from plasticc.training import process_meta

np.warnings.filterwarnings('ignore')
gc.enable()


def save_importances(importances_):
    mean_gain = importances_[['gain', 'feature']].groupby('feature').mean()
    importances_['mean_gain'] = importances_['feature'].map(mean_gain['gain'])
    return importances_


def process_test(
        clfs,  # List of classifiers
        features,
        featurize_configs,
        train_mean,
        output_path='predictions.csv',
        meta_path='data/raw/test_set_metadata.csv',
        test_path='data/raw/test_set.csv',
        id_colname='object_id',
        chunks=5000000
):
    start = time.time()

    meta_test = process_meta(meta_path)
    # meta_test.set_index(id_colname,inplace=True)

    remain_df = None
    for i_c, df in enumerate(pd.read_csv(test_path, chunksize=chunks, iterator=True)):
        # Check object_ids
        # I believe np.unique keeps the order of group_ids as they appear in the file
        unique_ids = np.unique(df[id_colname])

        new_remain_df = df.loc[df[id_colname] == unique_ids[-1]].copy()
        if remain_df is None:
            df = df.loc[df[id_colname].isin(unique_ids[:-1])]
        else:
            df = pd.concat([remain_df, df.loc[df[id_colname].isin(unique_ids[:-1])]], axis=0)
        # Create remaining samples df
        remain_df = new_remain_df

        preds_df = predict_chunk(
            df_=df,
            clfs_=clfs,
            meta_=meta_test,
            features=features,
            featurize_configs=featurize_configs,
            train_mean=train_mean
        )

        if i_c == 0:
            preds_df.to_csv(output_path, header=True, mode='a', index=False)
        else:
            preds_df.to_csv(output_path, header=False, mode='a', index=False)

        del preds_df
        gc.collect()
        print('{:15d} done in {:5.1f} minutes' .format(
                chunks * (i_c + 1), (time.time() - start) / 60), flush=True)

    # Compute last object in remain_df
    preds_df = predict_chunk(
        df_=remain_df,
        clfs_=clfs,
        meta_=meta_test,
        features=features,
        featurize_configs=featurize_configs,
        train_mean=train_mean
    )
    preds_df.to_csv(output_path, header=False, mode='a', index=False)
    return
