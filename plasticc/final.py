import gc
import time

import numpy as np
import pandas as pd

from plasticc.training import predict_chunk
from plasticc.training import process_meta

np.warnings.filterwarnings('ignore')
gc.enable()


def process_test(clfs,
                 features,
                 featurize_configs,
                 train_mean,
                 filename='predictions.csv',
                 chunks=5000000):
    start = time.time()

    meta_test = process_meta('../data/raw/test_set_metadata.csv')
    # meta_test.set_index('object_id',inplace=True)

    remain_df = None
    for i_c, df in enumerate(pd.read_csv('../data/raw/test_set.csv', chunksize=chunks, iterator=True)):
        # Check object_ids
        # I believe np.unique keeps the order of group_ids as they appear in the file
        unique_ids = np.unique(df['object_id'])

        new_remain_df = df.loc[df['object_id'] == unique_ids[-1]].copy()
        if remain_df is None:
            df = df.loc[df['object_id'].isin(unique_ids[:-1])]
        else:
            df = pd.concat([remain_df, df.loc[df['object_id'].isin(unique_ids[:-1])]], axis=0)
        # Create remaining samples df
        remain_df = new_remain_df

        preds_df = predict_chunk(df_=df,
                                 clfs_=clfs,
                                 meta_=meta_test,
                                 features=features,
                                 featurize_configs=featurize_configs,
                                 train_mean=train_mean)

        if i_c == 0:
            preds_df.to_csv(filename, header=True, mode='a', index=False)
        else:
            preds_df.to_csv(filename, header=False, mode='a', index=False)

        del preds_df
        gc.collect()
        print('{:15d} done in {:5.1f} minutes' .format(
                chunks * (i_c + 1), (time.time() - start) / 60), flush=True)

    # Compute last object in remain_df
    preds_df = predict_chunk(df_=remain_df,
                             clfs_=clfs,
                             meta_=meta_test,
                             features=features,
                             featurize_configs=featurize_configs,
                             train_mean=train_mean)

    preds_df.to_csv(filename, header=False, mode='a', index=False)
    return
