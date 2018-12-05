import pickle
from typing import List

from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


random_seed = 2222


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
        nthread=4,
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
        n_jobs=4,
        random_state=random_seed
    )
