from xgboost import XGBClassifier
import pickle
from plasticc.dataset import Dataset


def train(dataset: Dataset,
          output_path: str,
          xgb_params={},
          yname="target") -> XGBClassifier:
    dataset = dataset.train_df(y=yname)

    X = dataset.X
    Y = dataset.y

    model = XGBClassifier(**xgb_params)
    model.fit(X, Y)

    if output_path is not None:
        pickle.dump(model, open(output_path, "wb"))
    return model
