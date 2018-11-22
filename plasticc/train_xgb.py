from xgboost import XGBClassifier
import pickle
from plasticc.dataset import Dataset


def train(dataset: Dataset,
          output_path: str,
          xgb_params={},
          yname="target") -> XGBClassifier:

    X, y = dataset.train

    cons = X.join(y)
    cons.dropna(inplace=True)
    X, y = cons[cons.columns[:-1]], cons[cons.columns[-1]]

    model = XGBClassifier(**xgb_params)
    model.fit(X.values, y.values)

    if output_path is not None:
        pickle.dump(model, open(output_path, "wb"))
    return model
