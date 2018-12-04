import pickle
from typing import List

from xgboost import XGBClassifier
import numpy as np
import pandas as pd

from plasticc.dataset import Dataset, MultiDataset
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split


random_seed = 2222


def train_model(
        dataset_name: str,
        output_path: str,
        model_name: str,
        yname="target"
):

    model = resolve_model_name(model_name)
    dataset = resolve_dataset_name(dataset_name)

    X, y = dataset.train

    X.fillna(0, inplace=True)
    assert(X.notna().all().all())
    X.drop(columns=[col for col in set(X.columns) if col.endswith('_meta')],
           inplace=True)

    print("Before infinity removal:", X.shape)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    na_cols = null_values(X)
    X.drop(columns=na_cols, inplace=True)
    print("After infinity removal:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15,
                                                        random_state=42)

    print("Training!")
    model.fit(X_train.values.astype(np.float32), y_train.values.astype(np.int))
    print("Done.")
    _save_model_if_path_not_none(model, output_path)
    return model


def resolve_dataset_name(name):
    if name == 'simple-2':
        return Dataset('data/sets/simple-12-01/', y_colname='target')
    elif name == 'simple':
        return Dataset('data/sets/simple/', y_colname='target')
    elif name == 'tsfresh':
        return Dataset('data/sets/tsfresh-kaggle-sample/', y_colname='target')
    elif name == 'tsfresh-simple':
        return MultiDataset(['data/sets/tsfresh-kaggle-sample/', 'data/sets/simple-kaggle-sample'],
                            y_colname='target')
    else:
        raise Exception("No such dataset registered")


def resolve_model_name(name):
    if name == "xgb":
        return build_xgb()
    elif name == "bagged_xgb":
        return build_bagged_model(build_xgb)
    else:
        raise Exception(f"Unknown model: {name}")


def null_values(X: pd.DataFrame) -> List[str]:
    print("Total columns:", len(X.columns))
    na_cols = [col for col in X.columns if X[col].isna().any()]
    print("Total NA columns: ", len(na_cols))
    if len(na_cols) < 10:
        print("NA values by column:")
        print({na_col: X[na_col].isna().sum() for na_col in na_cols})
    return na_cols


def build_xgb():
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
        nthread=-1
    )

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


def _save_model_if_path_not_none(model, path):
    if path is not None:
        pickle.dump(model, open(path, "wb"))
    print(f"Saved model to {path}")
    return model
