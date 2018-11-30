from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.metrics import make_scorer
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
    """ Wrapping loss function for scikit learn """
    return make_scorer(score_func=kaggle_loss, greater_is_better=False)

def kaggle_loss(y, y_pred, **kwargs):
    """ Implemented multi log loss from kaggle competition """
    #Analised weights for classes
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

    #evading log extremes
    min_values = (np.ones(len(y_pred)*len(weight_dict))*(1-1e-15)).reshape(len(y_pred),len(weight_dict))
    removed_ones = np.minimum(y_pred,min_values)
    removed_zeros = np.maximum(np.minimum(y_pred,min_values), (1e-15))
    y_pred = removed_zeros

    #calculating loss
    ln_p = np.log(y_pred)
    N = np.array(np.sum(y, axis=0))
    weights = np.array(list(weight_dict.values()))
    multiplied_weights = np.multiply(weights,np.multiply(y,ln_p)/N)
    #removing nan in case of not represented classes (in that case Ni equals 0 and we get division by zero)
    nan_removed = np.nan_to_num(multiplied_weights)
    sum_from_j_to_Ni = np.sum(nan_removed, axis=1)
    sum_from_i_to_M = np.sum(sum_from_j_to_Ni)
    numerator = sum_from_i_to_M
    denominator = sum(weight_dict.values())*magic_const
    return -numerator/denominator
