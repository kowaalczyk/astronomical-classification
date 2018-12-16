import gc
import time
from typing import List
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
from tqdm import tqdm

from plasticc.featurize import process_meta, featurize

np.warnings.filterwarnings('ignore')
gc.enable()


def featurize_test(featurize_configs,
                   n_jobs,
                   meta_path,
                   test_path,
                   output_path,
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

        # process all features
        full_test = featurize(
            df, meta_test,
            featurize_configs['aggs'],
            featurize_configs['fcp'],
            n_jobs=n_jobs
        )
        full_test.fillna(0, inplace=True)
        if i_c == 0:
            full_test.to_csv(output_path, header=True, mode='w', index=False)
        else:
            full_test.to_csv(output_path, header=False, mode='a', index=False)

        del full_test
        gc.collect()

        rows_complete, time_complete = chunks * (i_c + 1), (time.time() - start) / 60
        print(f'{rows_complete:15d} done in {time_complete:5.1f} minutes', flush=True)

    full_test = featurize(
        remain_df,
        meta_test,
        featurize_configs['aggs'],
        featurize_configs['fcp'],
        n_jobs=n_jobs
    )
    full_test.fillna(0, inplace=True)
    full_test.to_csv(output_path, header=False, mode='a', index=False)


def predict_test(
        input_path: str,
        output_path: str,
        feature_colnames: List[str],
        id_colname: str,
        clfs: List,  # list of classifiers
        class_99_target_mean=0.12,  # class_99 will be scaled to match this mean value
        verbose: bool=False
) -> pd.DataFrame:
    if 'object_id' in feature_colnames:
        raise KeyError('Cannot use object_id as a feature!')
    print("Loading data...")
    X_test = pd.read_csv(input_path, index_col=id_colname)

    print("Generating predictions...")
    subm = predict_chunk(X=X_test, features=feature_colnames, clfs=clfs, verbose=verbose)

    print("Postprocessing...")
    # bagging - compute mean prediction from all clfs for each object_id
    print(f"Submission shape before grouping: {subm.shape}")
    subm_single = subm.groupby('object_id').mean()
    print(f"Submission shape after grouping: {subm_single.shape}")
    # scale class_99 mean to match the desired value before normalizing other classes
    class_99_mean = subm_single['class_99'].mean()
    print(f"Class 99 mean probability before normalization: {class_99_mean}")
    scaling_factor = class_99_target_mean / class_99_mean
    subm_single['class_99'] = subm_single['class_99'] * scaling_factor
    print(f"Class 99 mean probability after normalization: {subm_single['class_99'].mean()}")
    # normalization - all classes' probabilities should be equal to 1.0 for each object_id
    subm_sum = subm_single.sum(axis=1)
    for col in subm_single.columns:
        subm_single[col] = subm_single[col] / subm_sum
    max_err = np.max(np.abs(subm_single.sum(axis=1).values - 1.0))
    if max_err > 1e-15:
        print(f"Warning: high error in submission normalization: {max_err}")
    # round all predictions to float32 so that CSV is twice smaller (Kaggle uses 32-bit accuracy to calculate score anyway)
    subm_single = subm_single.round(decimals=8).astype(np.float32)
    # make sure index is typed correctly
    subm_single.index = subm_single.index.astype(np.int)
    print(f"Submission shape after postprocessing: {subm_single.shape}")

    print('Validating submission file...')
    if not subm_single.shape == (3492890, 15):
        print("Invalid shape")
        return subm_single
    if not subm_single.index.dtype == np.int:
        print("Invalid index")
        return subm_single

    if output_path is not None:
        print("Saving submission...")
        try:
            subm_single.to_csv(output_path, index=True)
            print(f"Submission saved to f{output_path}")
        except Exception as e:
            print(e)
    return subm_single


def reny_entropy(alpha, vector):
    if alpha < 0 or alpha == 1:
        raise Exception('alpha must not be 1 neither negative')

    coeff = 1 / (1 - alpha)
    powered = vector ** alpha
    sumpow = np.sum(powered, axis=1)
    logof = np.log(sumpow) / np.log(alpha)
    return logof * coeff


def predict_chunk(X, clfs, features, verbose=False):
    # Make predictions
    preds_ = None
    if verbose:
        for (clf, scor) in tqdm(clfs):  # display progressbar
            if preds_ is None:
                preds_ = clf.predict_proba(X[features], num_iteration=clf.best_iteration_)
            else:
                preds_ += clf.predict_proba(X[features], num_iteration=clf.best_iteration_)
    else:
        for clf in clfs:
            if preds_ is None:
                preds_ = clf.predict_proba(X[features], num_iteration=clf.best_iteration_)
            else:
                preds_ += clf.predict_proba(X[features], num_iteration=clf.best_iteration_)
    preds_ = preds_ / len(clfs)

    preds_99 = np.ones(preds_.shape[0])
    for i in range(preds_.shape[1]):
        preds_99 *= (1 - preds_[:, i])

    # Create DataFrame from predictions
    preds_df_ = pd.DataFrame(preds_,
                             columns=['class_{}'.format(s) for s in clfs[0][0].classes_])

    preds_df_['object_id'] = X.index  # when the dataframe is loaded with index_col=object_id there is no such column as object_id
    preds_df_['class_99'] = preds_99
    return preds_df_
