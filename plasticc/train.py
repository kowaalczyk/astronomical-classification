from typing import NamedTuple

from plasticc.dataset import Dataset

import numpy as np
from tqdm.autonotebook import tqdm


random_state = 42


class ModelWithScore(NamedTuple):
    model: object=None
    score: float=0.0


def train_model(model, dataset: Dataset, cv_strategy):
    """ See plasticc.resolvers for available cv_splits (cv strategies) """
    X, y = dataset.train
    assert(X.notna().all().all())
    
    X_val, y_val = X.values.astype(np.float32), y.values.astype(np.int)
    splits = cv_strategy.split(X_val, y_val)
    n_splits = cv_strategy.get_n_splits(X_val, y_val)
    
    best_model = ModelWithScore()
    for train_index, test_index in tqdm(splits, total=n_splits):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]
        current_model = _train_and_score(model, X_train, y_train, X_test, y_test)
        if current_model.score > best_model.score:
            best_model = current_model  # TODO: Consider saving all models to make a bagging ensemble
    return best_model.model


def _train_and_score(model, X_train, y_train, X_test, y_test) -> ModelWithScore:
    try:
        model.fit(X_train.values.astype(np.float32), y_train.values.astype(np.int), verbose=True)
    except Exception:
        model.fit(X_train.values.astype(np.float32), y_train.values.astype(np.int))
    accuracy = model.score(X_test.values.astype(np.float32), y_test.values.astype(np.int))
    print(f"Accuracy: {accuracy}")
    return ModelWithScore(model, accuracy)
