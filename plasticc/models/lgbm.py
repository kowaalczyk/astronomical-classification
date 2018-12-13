import gc

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

from plasticc.models.utils import multi_weighted_logloss, build_importance_df


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


def lgbm_multi_weighted_logloss(y_true, y_preds):
    """
    refactor from
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # Taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weights = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}

    loss = multi_weighted_logloss(y_true, y_preds, classes, class_weights)
    return 'wloss', loss, False


def lgbm_modeling_cross_validation(
        params: dict,
        X_features,
        y,
        classes,  # List of class names
        class_weights,  # Dict class -> weight:int
        nr_fold=5,
        random_state=1,
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

    oof_preds = np.zeros((len(X_features), np.unique(y).shape[0]))
    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
        trn_x, trn_y = X_features.iloc[trn_], y.iloc[trn_]
        val_x, val_y = X_features.iloc[val_], y.iloc[val_]

        sm = SMOTE(k_neighbors=7, n_jobs=params['n_jobs'], random_state=42)
        trn_x, trn_y = sm.fit_resample(trn_x, trn_y)
        trn_x, trn_y = pd.DataFrame(trn_x, columns=X_features.columns), pd.Series(trn_y)

        clf = LGBMClassifier(**params)
        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            eval_metric=lgbm_multi_weighted_logloss,
            verbose=100,
            early_stopping_rounds=50,
            sample_weight=trn_y.map(weights)
        )

        scor = multi_weighted_logloss(
                val_y,
                oof_preds[val_, :],
                classes,
                class_weights
            )

        clfs.append((clf, scor))

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
    importances = build_importance_df(importances_=importances)
    return clfs, score, importances
