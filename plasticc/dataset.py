import os

import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm


class DatasetException(Exception):
    pass


class Dataset(object):
    """
    Dataset wrapper for easier path management in scripts.
    Provides access to train set CSV and iteration over test set CSVs.
    If dataset does not have denormalized metadata 
    (= metadata is in separate CSVs like in the raw data),
    the class provides access to metadata CSVs.
    """
    def __init__(self, path):
        self.path = path

    def has_meta(self):
        return os.path.exists(os.path.join(self.path, 'meta/'))

    @classmethod
    def with_structure(dataset_class, path, has_meta=False):
        """
        Creates an empty dataset directory structure in provided path and 
        returns a dataset wrapper for this path.
        """
        os.makedirs(os.path.join(path, 'test/'))
        if has_meta:
            os.makedirs(os.path.join(path, 'meta/'))
        return dataset_class(path)

    @property
    def meta_path(self, csv_name):
        if not self.has_meta():
            raise DatasetException("Dataset has no metadata!")
        return os.path.join(self.path, 'meta/', csv_name)

    @property
    def train_meta_df(self):
        return pd.read_csv(self.meta_path('train.csv'))
 
    @property
    def test_meta_df(self):
        return pd.read_csv(self.meta_path('test.csv'))

    @property
    def train_path(self):
        return os.path.join(self.path, 'train.csv')

    @property
    def test_paths(self):
        test_path = os.path.join(self.path, 'test/')
        test_file_names = sorted(os.listdir(test_path))
        test_file_paths = [os.path.join(test_path, f) for f in test_file_name]
        return test_file_paths

    @property
    def train_df(self):
        return pd.read_csv(self.train_path)

    def iter_test_dfs(self):
        for csv_name in self.test_paths:
            yield pd.read_csv(csv_name)


def batch_data(
        ts_reader: pd.io.parsers.TextFileReader, 
        meta_df: pd.DataFrame=None, 
        output_dir='../data/sets/base/test/', 
        lines=453653105
    ):
    """
    Splits pd.DataFrame iterated with ts_reader into batches.
    Each batch is saved as csv file in output_dir.
    If meta_df is provided, each record in time series is joined 
    with corresponding object's metadata (making the process ~3x slower).
    Lines argument is necessary for progressbar to work correctly.
    """
    reminder_df = pd.DataFrame()
    with tqdm(total=lines) as progressbar:
        for batch in ts_reader:
            if meta_df is not None:
                batch = batch.join(
                    meta_df, 
                    on='object_id', 
                    rsuffix='meta_', 
                    sort=True
                )
            # prepend reminder of rows from previous iteration
            if len(reminder_df) > 0:
                current_df = pd.concat([reminder_df, batch], axis=0)
            else:
                current_df = batch
            current_df.sort_values(by=['object_id', 'mjd', 'passband'], inplace=True)
            # separate reminder for next iteration and save current batch
            last_id = current_df.iloc[-1]['object_id']
            reminder_df = current_df[current_df['object_id'] == last_id]
            save_df = current_df[current_df['object_id'] != last_id]
            _save_batch(save_df, output_dir)
            progressbar.update(len(save_df))
        # save last reminder
        if len(reminder_df) > 0:
            _save_batch(reminder_df, output_dir)
            progressbar.update(len(reminder_df))


def _save_batch(batch: pd.DataFrame, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_filename = _name_for_batch(batch)
    batch.to_csv(os.path.join(output_dir, save_filename))


def _name_for_batch(batch: pd.DataFrame, pad_to_length=12):
    min_id_str = str(batch['object_id'].min()).zfill(pad_to_length)
    max_id_str = str(batch['object_id'].max()).zfill(pad_to_length)
    return f"test-batch-{min_id_str}-{max_id_str}.csv"
