import gc

import numpy as np
import pandas as pd

gc.enable()
np.warnings.filterwarnings('ignore')


def build_importance_df(importances_):
    mean_gain = importances_[['gain', 'feature']].groupby('feature').mean()
    importances_['mean_gain'] = importances_['feature'].map(mean_gain['gain'])
    return importances_


def multi_weighted_logloss(y_true, y_preds, classes, class_weights):
    """
    refactor from
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')
    # Trasform y_true in dummies
    # Don't know why y_true sometimes comes as numbers of classes or as values from 0 to 13
    # It just works now 
    if np.max(y_true) <= 13:
        y_ohe_with_bonus_rows = pd.get_dummies(np.concatenate((np.array(list(range(14))),y_true)))
    else:
        y_ohe_with_bonus_rows = pd.get_dummies(np.concatenate((np.array(classes),y_true)))
    y_ohe = y_ohe_with_bonus_rows[14:]
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weights[k] for k in sorted(class_weights.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    # sometimes, when we get no representatives of some class, we get nans
    y_w = np.nan_to_num(y_w)
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss
