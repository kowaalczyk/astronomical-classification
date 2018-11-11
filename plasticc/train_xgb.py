from xgboost import XGBClassifier
import pickle
from plasticc.dataset import Dataset
import math
import numpy as np
import pandas as pd

def train(dataset: Dataset,
          output_path: str,
          xgb_params={},
          yname="target") -> XGBClassifier:

    X, y = dataset.train

    nans = []
    for i in range(len(y.values)):
        if math.isnan(y.values[i]) or np.isnan(y.values[i]):
            nans.append(i)

    X.drop(nans, 0, inplace=True)
    y.drop(nans, 0, inplace=True)

    model = XGBClassifier(**xgb_params)
    model.fit(X.values, y.values)

    if output_path is not None:
        pickle.dump(model, open(output_path, "wb"))
    return model
