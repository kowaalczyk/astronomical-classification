import pandas as pd

from plasticc.dataset import Dataset, save_batch


def from_base(base_dataset: Dataset, out_dataset_path: str, process_test=True):
    out_dataset = Dataset.with_structure(out_dataset_path, has_meta=False)
    train_out_df = _extract_features(
        base_dataset.train_df, 
        base_dataset.train_meta_df
    )
    # TODO


def _extract_features(series_df: pd.DataFrame, meta_df: pd.DataFrame):
    pass  # TODO
