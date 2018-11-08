import os
from typing import List

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

    def __init__(self, path: str):
        self.path = path
        self._data = None
        self._X = None
        self._y = None

    @property
    def values(self) -> pd.DataFrame:
        return self._data

    @property
    def X(self) -> pd.DataFrame:
        return self._X

    @property
    def y(self) -> pd.DataFrame:
        return self._y

    @property
    def train_path(self) -> str:
        return os.path.join(self.path, 'train.csv')

    @property
    def test_path(self) -> str:
        return os.path.join(self.path, 'test/')

    @property
    def test_paths(self) -> List[str]:
        test_file_names = sorted(os.listdir(self.test_path))
        test_file_paths = [os.path.join(self.test_path, f) for f in test_file_names]
        return test_file_paths

    @property
    def iter_test_dfs(self):
        for csv_name in tqdm(self.test_paths):
            yield Dataset(csv_name).train_df()

    def has_meta(self):
        return os.path.exists(os.path.join(self.path, 'meta/'))

    def meta_path(self, csv_name: str) -> str:
        if not self.has_meta():
            raise DatasetException("Dataset has no metadata!")
        return os.path.join(self.path, 'meta/', csv_name)

    def train_meta_df(self, y=None):
        if self._data is not None:
            raise DatasetException("This dataset has already set data!")
        if y is None:
            raise DatasetException("You have to specify y column name in dataset!")
        self._data = pd.read_csv(self.meta_path('train.csv'))
        self._data.index = self._data['object_id']
        if y not in self._data.columns:
            raise DatasetException("Specified y column name does not exist in dataset!")
        self._X = self._data.loc[:, self.data.columns != y]
        self._y = self._data.loc[:, self.data.columns == y]
        return self

    def test_meta_df(self) -> pd.DataFrame:
        if self._data is not None:
            raise DatasetException("This dataset has already set data!")
        self._data = pd.read_csv(self.meta_path('test.csv'))
        self._data.index = self._data['object_id']
        self._X = self._data
        return self

    def train_df(self, y=None):
        if self._data is not None:
            raise DatasetException("This dataset has already set data!")
        if y is None:
            raise DatasetException("You have to specify y column name in dataset!")
        self._data = pd.read_csv(self.train_path)
        self._data.index = self._data['object_id']
        if y not in self._data.columns:
            raise DatasetException("Specified y column name does not exist in dataset!")
        self._X = self._data.loc[:, self._data.columns != y]
        self._y = self._data.loc[:, self._data.columns == y]
        return self


class Dataset_manager(object):
    """
    Static class creating a dataset directory structure and an instance
    """
    @staticmethod
    def get_dataset_with_structure(path: str, has_meta=False):
        """
        Creates an empty dataset directory structure in provided path and
        returns a Dataset for this path.
        """
        os.makedirs(os.path.join(path, 'test/'))
        if has_meta:
            os.makedirs(os.path.join(path, 'meta/'))
        return Dataset(path)


def batch_data(
        ts_reader: pd.io.parsers.TextFileReader,
        meta_df: pd.DataFrame = None,
        output_dir='../data/sets/base/test/',
        lines=453653105):
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


def _name_for_batch(batch: pd.DataFrame, pad_to_length=12) -> str:
    min_id_str = str(batch['object_id'].min()).zfill(pad_to_length)
    max_id_str = str(batch['object_id'].max()).zfill(pad_to_length)
    return f"test-batch-{min_id_str}-{max_id_str}.csv"
