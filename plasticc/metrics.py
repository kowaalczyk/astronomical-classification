from sklearn.model_selection import ShuffleSplit, cross_val_score
from xgboost import XGBClassifier
import pandas as pd
import numpy as np


def xgb_score(dataset_path: str,
              cv_params: dict,
              xgb_params={},
              scoring="f1_macro") -> np.ndarray:
    dataset = pd.read_csv(dataset_path, delimiter=',')

    X = dataset[dataset.columns[:-1]]
    Y = dataset[dataset.columns[-1]]
    model = XGBClassifier(**xgb_params)
    cv = ShuffleSplit(**cv_params)
    scores = cross_val_score(model, X, Y, cv=cv, scoring=scoring)
    return scores
