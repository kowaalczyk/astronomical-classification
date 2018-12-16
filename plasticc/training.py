import gc
import os
from datetime import datetime as dt
from typing import NamedTuple, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from plasticc.models import lgbm_modeling_cross_validation, xgb_modeling_cross_validation, mlp_modeling_cross_validation


np.warnings.filterwarnings('ignore')
gc.enable()


def path_from_cv_score(score, submissions_directory: str=os.path.abspath('../../submissions'), suffix: str='') -> str:
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
        model: str = 'lgbm',  # 'lgbm' or 'xgb' or 'mlp' or 'both_gbs'
        model_params: dict = {},
        fit_params: dict={},  # only for mlp model: epochs, batch_size & verbose
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
            params=model_params['lgbm'],
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
            params=model_params['xgb'],
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
    elif model == 'mlp':
        clfs, score = mlp_modeling_cross_validation(
            X_features=X_features,
            y=y,
            classes=classes,
            class_weights=weights,
            params=model_params['mlp'],
            fit_params=fit_params,
            nr_fold=nr_fold,
            random_state=random_state
        )
        return TrainingResult(
            clfs=clfs,
            score=score,
            importances=None
        )
    elif model == 'both_gbs':
        xclfs, xscore, ximp = train_and_validate(
            X=X,
            y=y,
            feature_colnames=feature_colnames,
            id_colname=id_colname,
            model='xgb',
            model_params=model_params,
            nr_fold=nr_fold,
            random_state=random_state
        )
        lclfs, lscore, limp = train_and_validate(
            X=X,
            y=y,
            feature_colnames=feature_colnames,
            id_colname=id_colname,
            model='lgbm',
            model_params=model_params,
            nr_fold=nr_fold,
            random_state=random_state
        )
        return TrainingResult(
            clfs=xclfs + lclfs,
            score=(lscore + xscore)/2,
            importances=pd.concat([ximp, limp])
        )
    else:
        raise ValueError("Unknown model")


class SearchResult(NamedTuple):
    training_results: List[TrainingResult]
    model_params: List[dict]
    best_idx: int


def random_search(
        n_iter: int,
        X: pd.DataFrame,
        y: pd.Series,
        feature_colnames: List[str],
        id_colname: str='object_id',
        model: str='lgbm',  # 'lgbm' or 'xgb'
        search_params: dict={},
        nr_fold: int=5,
        random_state: int=1,
) -> Tuple[List[TrainingResult], int]:
    input_params = [None for _ in range(n_iter)]
    training_results = [None for _ in range(n_iter)]
    best_idx = -1
    for i in tqdm(range(n_iter)):
        input_params[i] = {
            key: np.random.choice(search_params[key]) for key in search_params.keys()
        }
        training_results[i] = [c for (c, _) in train_and_validate(
            X=X,
            y=y,
            feature_colnames=feature_colnames,
            id_colname=id_colname,
            model=model,
            model_params=input_params[i],
            nr_fold=nr_fold,
            random_state=random_state
        )]
        if best_idx == -1 or training_results[i].score < training_results[best_idx].score:
            best_idx = i
    return SearchResult(
        training_results=training_results,
        model_params=input_params,
        best_idx=best_idx
    )
