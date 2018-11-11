from sklearn.model_selection import ShuffleSplit, cross_val_score
from xgboost import XGBClassifier
import numpy as np
from plasticc.dataset import Dataset
import math


def xgb_score(dataset: Dataset,
              cv_params: dict,
              xgb_params={},
              scoring="f1_macro") -> np.ndarray:

    X, y = dataset.train
    X = X.values
    y = y.values
    nans = []
    for i in range(len(y)):
        if math.isnan(y[i]) or np.isnan(y[i]):
            nans.append(i)


    X = np.delete(X, nans, 0)
    y = np.delete(y, nans, 0)
    print (y)

    model = XGBClassifier(**xgb_params)
    cv = ShuffleSplit(**cv_params)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, error_score='raise')
    return scores
