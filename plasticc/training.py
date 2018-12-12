import gc
import os
from datetime import datetime as dt
from typing import NamedTuple, List

import numpy as np
import pandas as pd

from plasticc.models import lgbm_modeling_cross_validation, xgb_modeling_cross_validation


np.warnings.filterwarnings('ignore')
gc.enable()


def path_from_cv_score(score, submissions_directory: str=os.path.abspath('../submissions'), suffix: str='') -> str:
    filename = f"subm_{score:.6f}_{dt.now().strftime('%Y-%m-%d-%H-%M')}{suffix}.csv"
    return os.path.join(submissions_directory, filename)


class TrainingResult(NamedTuple):
    clfs: List
    score: float
    importances: pd.DataFrame


def classes_and_weights(y: pd.Series):
    classes = sorted(y.unique())
    class_weights = {c: 1 for c in classes}
    class_weights.update({c: 2 for c in [64, 15]})
    return classes, class_weights


def train_and_validate(
        X: pd.DataFrame,
        y: pd.Series,
        feature_colnames: List[str],
        id_colname: str = 'object_id',
        model: str = 'lgbm',  # 'lgbm' or 'xgb' or both
        model_params: dict = {},
        nr_fold: int = 5,
        random_state: int = 1
) -> TrainingResult:
    assert(id_colname not in feature_colnames)
    X_features = X[feature_colnames]
    X_features.index = X[id_colname]

    classes, weights = classes_and_weights(y)

    if model == 'lgbm':
        clfs, score, importances = lgbm_modeling_cross_validation(
            X_features=X_features,
            y=y,
            params=model_params,
            classes=classes,
            class_weights=weights,
            nr_fold=nr_fold,
            random_state=random_state
        )
        return TrainingResult(
            clfs=clfs,
            score=score,
            importances=importances
        )
    elif model == 'xgb':
        clfs, score, importances = xgb_modeling_cross_validation(
            X_features=X_features,
            y=y,
            params=model_params,
            classes=classes,
            class_weights=weights,
            nr_fold=nr_fold,
            random_state=random_state
        )
        return TrainingResult(
            clfs=clfs,
            score=score,
            importances=importances
        )

    elif model == 'both':
        xclfs, xscore, ximp = train_and_validate(X=X,
                                                 y=y,
                                                 feature_colnames=feature_colnames,
                                                 id_colname=id_colname,
                                                 model='xgb',
                                                 model_params=model_params,
                                                 nr_fold=nr_fold,
                                                 random_state=random_state)
        lclfs, lscore, limp = train_and_validate(X=X,
                                                 y=y,
                                                 feature_colnames=feature_colnames,
                                                 id_colname=id_colname,
                                                 model='lgbm',
                                                 model_params=model_params,
                                                 nr_fold=nr_fold,
                                                 random_state=random_state)
        return TrainingResult(clfs=xclfs + lclfs,
                              score=(lscore + xscore)/2,
                              importances=pd.concat([ximp, limp]))
    else:
        raise ValueError("Unknown model, must be either 'both', 'xgb' or 'lgbm'")
