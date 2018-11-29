import pickle

from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split

from plasticc.dataset import Dataset


random_seed = 2222


def train_model(
        dataset: Dataset,
        output_path: str,
        model,
        yname="target"
):
    X, y = dataset.train
    model.fit(X.values, y.values)
    _save_model_if_path_not_none(model, output_path)
    return model


def build_xgb():
    pass  # TODO


def build_bagged_model(base_estimator):
    return BaggingClassifier(
        base_estimator=base_estimator
        n_estimators=20, 
        max_samples=0., 
        max_features=0.67, 
        bootstrap=True, 
        bootstrap_features=True, 
        oob_score=True, 
        n_jobs=-1, 
        random_state=random_seed
    )

# train_model(build_bagged_model(build_xgb()))

def train_bagging_model(
        dataset: Dataset,
        output_path: str,
        base_estimator,
        test_frac: float=0.1
):
    X, y = dataset.train
    X_train, X_test, y_train, y_test = train_test_split(test_size=test_frac, random_state=random_seed)
    
    bc = build_bagged_model(base_estimator=build_xgb())
    bc.fit(X_train.values.astype(np.float32), y_train.values.astype(np.int))
    
    print("Accuracy score on validation set: ", bc.score(X_test, y_test))
    _save_model_if_path_not_none(bc, output_path)

def _save_model_if_path_not_none(model, path):
    if output_path is not None:
        pickle.dump(model, open(output_path, "wb"))
    return model
