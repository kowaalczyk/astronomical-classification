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

def kaggle_scoring():
    return make_scorer(score_func=kaggle_loss, greater_is_better=False)

def kaggle_loss(y, y_pred, **kwargs):
    magic_const = 1.9188
    weight_dict = {
      "class_6": 1,
      "class_15": 2,
      "class_16": 1,
      "class_42": 1,
      "class_52": 1,
      "class_53": 1,
      "class_62": 1,
      "class_64": 2,
      "class_65": 1,
      "class_67": 1,
      "class_88": 1,
      "class_90": 1,
      "class_92": 1,
      "class_95": 1,
      "class_99": 2,
    }
    y_pred = np.maximum(np.minimum(y_pred,(np.ones(len(y_pred)*len(weight_dict))*(1-1e-15)).reshape(len(y_pred),len(weight_dict))),(1e-15))
    return -np.sum(np.sum(np.nan_to_num(np.multiply(np.array(list(weight_dict.values())),np.multiply(y,np.log(y_pred))/np.array(np.sum(y, axis=0)))), axis=1)/sum(weight_dict.values())*magic_const)
    #return -(np.sum(np.multiply(np.array(weight_dict.values()),np.multiply(y,np.log(y_pred))/np.array(np.sum(y, axis=0))), axis=1))/sum(weight_dict.values())*magic_const
    
