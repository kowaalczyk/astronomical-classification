import gc

import numpy as np
import pandas as pd

from plasticc.featurize import featurize
from plasticc.metrics import haversine_plus

np.warnings.filterwarnings('ignore')
gc.enable()


def process_meta(filename):
    meta_df = pd.read_csv(filename)
    meta_dict = dict()
    # distance
    meta_dict.update(haversine_plus(meta_df['ra'].values, meta_df['decl'].values,
                                    meta_df['gal_l'].values, meta_df['gal_b'].values))

    meta_dict['hostgal_photoz_certain'] = np.multiply(
        meta_df['hostgal_photoz'].values,
        np.exp(meta_df['hostgal_photoz_err'].values))

    meta_df = pd.concat([meta_df, pd.DataFrame(meta_dict, index=meta_df.index)], axis=1)
    return meta_df


def predict_chunk(df_, clfs_, meta_, features, featurize_configs, train_mean):

    # process all features
    full_test = featurize(df_, meta_,
                          featurize_configs['aggs'],
                          featurize_configs['fcp'])
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
