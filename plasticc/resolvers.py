from plasticc import dataset as ds
from plasticc import models
from plasticc.cleaning_strategies import clean_kaggle_simple, clean_kaggle_no_pos

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


random_state = 5555


def resolve_dataset_name(name):
    if name == 'simple-2':
        return ds.Dataset('data/sets/simple-12-01/', y_colname='target')
    elif name == 'simple':
        return ds.Dataset('data/sets/simple/', y_colname='target')
    elif name == 'tsfresh':
        return ds.Dataset('data/sets/tsfresh-kaggle-sample/',
                          y_colname='target')
    elif name == 'kaggle-kernel':
        return ds.MultiDataset(
            ['data/sets/tsfresh-kaggle-sample/','data/sets/simple-kaggle-sample'],
            y_colname='target',
            cleaning_strategy=clean_kaggle_simple
        )
    elif name == 'kaggle-simple':
        return ds.MultiDataset(
            ['data/sets/simple-kaggle'],
            y_colname='target',
            cleaning_strategy=clean_kaggle_no_pos
        )
    else:
        raise Exception("No such dataset registered")


def resolve_model_name(name):
    if name == "xgb":
        return models.build_xgb()
    elif name == "bagged_xgb":
        return models.build_bagged_model(train.build_xgb())
    else:
        raise Exception(f"Unknown model: {name}")


def resolve_cv_strategy(name):
    if name == "stratified_kfold":
        return StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    elif name == "stratified_shuffle":
        return StratifiedShuffleSplit(n_splits=12, test_size=0.1, random_state=random_state)
