import gc
import time

import numpy as np
import pandas as pd

from plasticc.featurize import process_meta, featurize

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
        n_jobs,
        output_path='predictions.csv',
        meta_path='data/raw/test_set_metadata.csv',
        test_path='data/raw/test_set.csv',
        id_colname='object_id',
        chunks=5000000,
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
            train_mean=train_mean,
            n_jobs=n_jobs
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
        train_mean=train_mean,
        n_jobs=n_jobs
    )
    preds_df.to_csv(output_path, header=False, mode='a', index=False)
    return


def predict_chunk(df_, clfs_, meta_, features, featurize_configs, train_mean, n_jobs):

    # process all features
    full_test = featurize(df_, meta_,
                          featurize_configs['aggs'],
                          featurize_configs['fcp'],
                          n_jobs=n_jobs)
    full_test.fillna(0, inplace=True)

    # Make predictions
    preds_ = None
    for clf in clfs_:
        if preds_ is None:
            preds_ = clf.predict_proba(full_test[features])
        else:
            preds_ += clf.predict_proba(full_test[features])
    preds_ = preds_ / len(clfs_)

    # Compute preds_99 as the proba of class not being any of the others
    # preds_99 = 0.1 gives 1.769
    preds_99 = np.ones(preds_.shape[0])
    for i in range(preds_.shape[1]):
        preds_99 *= (1 - preds_[:, i])

    # Create DataFrame from predictions
    preds_df_ = pd.DataFrame(preds_,
                             columns=['class_{}'.format(s) for s in clfs_[0].classes_])
    preds_df_['object_id'] = full_test['object_id']
    preds_df_['class_99'] = 0.14 * preds_99 / np.mean(preds_99)
    return preds_df_
