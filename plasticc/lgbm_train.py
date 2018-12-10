import gc

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

from plasticc.metrics import (lgbm_multi_weighted_logloss,
                              multi_weighted_logloss)
from plasticc.training import build_importance_df

np.warnings.filterwarnings('ignore')
gc.enable()


best_params = {
    'device': 'cpu',
    'objective': 'multiclass',
    'num_class': 14,
    'boosting_type': 'gbdt',
    'n_jobs': 16,
    'max_depth': 7,
    'n_estimators': 1024,
    'subsample_freq': 2,
    'subsample_for_bin': 5000,
    'min_data_per_group': 100,
    'max_cat_to_onehot': 4,
    'cat_l2': 1.0,
    'cat_smooth': 59.5,
    'max_cat_threshold': 32,
    'metric_freq': 10,
    'verbosity': -1,
    'metric': 'multi_logloss',
    'xgboost_dart_mode': False,
    'uniform_drop': False,
    'colsample_bytree': 0.5,
    'drop_rate': 0.173,
    'learning_rate': 0.0267,
    'max_drop': 5,
    'min_child_samples': 10,
    'min_child_weight': 100.0,
    'min_split_gain': 0.1,
    'num_leaves': 7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.00023,
    'skip_drop': 0.44,
    'subsample': 0.75
}


def lgbm_modeling_cross_validation(
        params: dict,
        X,
        y,
        classes,  # List of class names
        class_weights,  # Dict class -> weight:int
        nr_fold=5,
        random_state=1,
        id_colname='object_id'
):
    # Compute weights
    w = y.value_counts()
    weights = {i: np.sum(w) / w[i] for i in w.index}

    clfs = []
    importances = pd.DataFrame()
    folds = StratifiedKFold(
        n_splits=nr_fold,
        shuffle=True,
        random_state=random_state
    )
    X_features = X[[col for col in X.columns if not col == id_colname]]
    
    oof_preds = np.zeros((len(X_features), np.unique(y).shape[0]))
    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
        trn_x, trn_y = X_features.iloc[trn_], y.iloc[trn_]
        val_x, val_y = X_features.iloc[val_], y.iloc[val_]

        clf = LGBMClassifier(**params)
        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            eval_metric=lgbm_multi_weighted_logloss,
            verbose=100,
            early_stopping_rounds=50,
            sample_weight=trn_y.map(weights)
        )
        clfs.append(clf)

        oof_preds[val_, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
        print('no {}-fold loss: {}'.format(
            fold_ + 1,
            multi_weighted_logloss(
                val_y, 
                oof_preds[val_, :],
                classes, 
                class_weights
        )))

        imp_df = pd.DataFrame({
                'feature': X_features.columns,
                'gain': clf.feature_importances_,
                'fold': [fold_ + 1] * len(X_features.columns),
                })
        importances = pd.concat([importances, imp_df], axis=0, sort=False)

    score = multi_weighted_logloss(
        y_true=y, 
        y_preds=oof_preds,
        classes=classes, 
        class_weights=class_weights
    )
    print('MULTI WEIGHTED LOG LOSS: {:.5f}'.format(score))
    df_importances = build_importance_df(importances_=importances)
    return clfs, score, importances
