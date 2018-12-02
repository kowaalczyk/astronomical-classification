import pickle
from typing import List

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

from plasticc.dataset import Dataset
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split

from plasticc.dataset import Dataset


random_seed = 2222


def train_model(
        dataset: Dataset,
        output_path: str,
        model,
        yname="target"
):
    X, y = dataset.train
    model.fit(X.values, y.values)
    _save_model_if_path_not_none(model, output_path)
    return model


def null_values(X: pd.DataFrame) -> List[str]:
    print("Total columns:", len(X.columns))
    na_cols = [col for col in X.columns if X[col].isna().any()]
    print("Total NA columns: ", len(na_cols))
    if len(na_cols) < 10:
        print("NA values by column:")
        print({na_col: X[na_col].isna().sum() for na_col in na_cols})
    return na_cols


def build_xgb(training_set: str):
    if training_set == 'simple-2':
        ds = Dataset('../data/sets/simple-12-01/', y_colname='target')
    elif training_set == 'simple':
        ds = Dataset('../data/sets/simple/', y_colname='target')
    elif training_set == 'tsfresh':
        ds = Dataset('../data/sets/tsfresh-sample/', y_colname='target')
    else:
        raise "No such dataset registered"

    X, y = ds.train

    X.fillna(0, inplace=True)
    assert(X.notna().all().all())
    X.drop(columns=[col for col in set(X.columns) if col.endswith('_meta')], inplace=True)

    print("Before infinity removal:", X.shape)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    na_cols = null_values(X)
    X.drop(columns=na_cols, inplace=True)
    print("After infinity removal:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    xgb_model = XGBClassifier(
        objective='multi:softmax',
        num_class=14,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.5,
        reg_alpha=0.01,
        reg_lambda=0.01,
        min_child_weight=10,
        n_estimators=1024,
        max_depth=3,
        nthread=16
    )

    xgb_model.fit(X_train, y_train, verbose=100, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='mlogloss', early_stopping_rounds=50)

    print(f"Score: {xgb_model.score(X_test, y_test)}")

    return xgb_model



def build_bagged_model(base_estimator):
    return BaggingClassifier(
        base_estimator=base_estimator,
        n_estimators=20,
        max_samples=0.,
        max_features=0.67,
        bootstrap=True,
        bootstrap_features=True,
        oob_score=True,
        n_jobs=-1,
        random_state=random_seed
    )

# train_model(build_bagged_model(build_xgb()))

def train_bagging_model(
        dataset: Dataset,
        output_path: str,
        base_estimator,
        test_frac: float = 0.1
):
    X, y = dataset.train
    X_train, X_test, y_train, y_test = train_test_split(test_size=test_frac, random_state=random_seed)

    bc = build_bagged_model(base_estimator=build_xgb())
    bc.fit(X_train.values.astype(np.float32), y_train.values.astype(np.int))

    print("Accuracy score on validation set: ", bc.score(X_test, y_test))
    _save_model_if_path_not_none(bc, output_path)

def _save_model_if_path_not_none(model, path):
    if path is not None:
        pickle.dump(model, open(path, "wb"))
    return model
