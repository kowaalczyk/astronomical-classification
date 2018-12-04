import plasticc.dataset as ds
from plasticc import train


def resolve_dataset_name(name):
    if name == 'simple-2':
        return ds.Dataset('data/sets/simple-12-01/', y_colname='target')
    elif name == 'simple':
        return ds.Dataset('data/sets/simple/', y_colname='target')
    elif name == 'tsfresh':
        return ds.Dataset('data/sets/tsfresh-kaggle-sample/',
                          y_colname='target')
    elif name == 'tsfresh-simple':
        return ds.MultiDataset(['data/sets/tsfresh-kaggle-sample/',
                                'data/sets/simple-kaggle-sample'],
                               y_colname='target')
    else:
        raise Exception("No such dataset registered")


def resolve_model_name(name):
    if name == "xgb":
        return train.build_xgb()
    elif name == "bagged_xgb":
        return train.build_bagged_model(train.build_xgb())
    else:
        raise Exception(f"Unknown model: {name}")
