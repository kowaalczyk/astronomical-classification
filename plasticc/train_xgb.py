from xgboost import XGBClassifier
import pickle
from plasticc.dataset import Dataset
import math
import numpy as np

def train(dataset: Dataset,
          output_path: str,
          xgb_params={},
          yname="target") -> XGBClassifier:

    X, y = dataset.train
    X = X.values
    y = y.values

    nans = []
    for i in range(len(y)):
        if math.isnan(y[i]) or np.isnan(y[i]):
            nans.append(i)


    X = np.delete(X, nans, 0)
    y = np.delete(y, nans, 0)



    model = XGBClassifier(**xgb_params)
    model.fit(X, y)

    if output_path is not None:
        pickle.dump(model, open(output_path, "wb"))
    return model
