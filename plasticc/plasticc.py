import pickle
import pandas as pd
from plasticc.dataset import Dataset


def load_model(model_path: str):
    return pickle.load(open(model_path, "rb"))


def prepare_submission(output_path: str, model, inp: Dataset):
    for d in inp.iter_test_dfs:
        sub = pd.DataFrame(model.predict(d).X)
        sub.to_csv(output_path + d.path, index=False)
