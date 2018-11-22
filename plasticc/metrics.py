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
#    X.dropna(inplace=True, axis=1)

#    cons = X.join(y)
#    cons.dropna(inplace=True)
#    X, y = cons[cons.columns[:-1]], cons[cons.columns[-1]]

    model = XGBClassifier(**xgb_params)
    cv = ShuffleSplit(**cv_params)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, error_score='raise')
    return scores
