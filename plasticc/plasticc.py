import pickle
import pandas as pd


def load_model(model_path: str):
    return pickle.load(open(model_path, "rb"))


def load_dataset(dataset_path: str) -> pd.core.frame.DataFrame:
    return pd.read_csv(dataset_path, delimiter=",")


def prepare_submission(output_path: str, model, inp: pd.core.frame.DataFrame):
    sub = pd.DataFrame(model.predict(inp))
    sub.to_csv(output_path, index=False)
