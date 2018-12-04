import os
from typing import List, Tuple

import pandas as pd
from tqdm.autonotebook import tqdm


class DatasetException(Exception):
    pass


class Dataset(object):
    """
    Dataset wrapper for easier path management in scripts.
    Provides access to training data if the y_column is passed on construction:
        ds = Dataset('path/to/dataset/', y='colname')
        X, y = ds.train  # pd.DataFrame, pd.Series
    Provides access to test data:
        ds = Dataset('path/to/dataset/')
        for df in ds.iter_test:
            df  # pd.DataFrame
    Dataset also exposes paths to train, test and metadata via its properties.
    """
    def __init__(self, path: str, y_colname: str=None, index_colname='object_id'):
        self.path = path
        self.y_colname = y_colname
        self.index_colname = index_colname

    @property
    def train(self) -> Tuple[pd.DataFrame, pd.Series]:
        """ Returns X, y for the training set. """
        if self.y_colname is None:
            raise DatasetException("Specify y_colname before training")
        train_df = self.train_indexed
        X_cols = [col for col in train_df.columns if not col == self.y_colname]
        return train_df[X_cols], train_df[self.y_colname].astype('category')

    @property
    def train_raw(self) -> pd.DataFrame:
        return pd.read_csv(self.train_path)
    
    @property
    def train_indexed(self) -> pd.DataFrame:
        return self.index(self.train_raw)

    @property
    def iter_test(self):
        """ Iterate over all of test set DataFrames. """
        for csv_path in tqdm(self.test_paths):
            yield pd.read_csv(csv_path)
            
    @property
    def iter_indexed_test(self):
        """ Iterate over all of test set DataFrames, with added indexes. """
        for csv_path in tqdm(self.test_paths):
            yield self.index(pd.read_csv(csv_path))

    @property
    def train_meta(self) -> pd.DataFrame:
        return self.index(pd.read_csv(self.meta_path('train.csv')))

    @property
    def test_meta(self) -> pd.DataFrame:
        return self.index(pd.read_csv(self.meta_path('test.csv')))

    def has_meta(self):
        return os.path.exists(os.path.join(self.path, 'meta/'))

    @property
    def train_path(self) -> str:
        """ Path to train CSV """
        return os.path.join(self.path, 'train.csv')

    @property
    def test_path(self) -> str:
        """ Path to test directory. """
        return os.path.join(self.path, 'test/')
    
    @property
    def test_filenames(self) -> List[str]:
        return sorted(os.listdir(self.test_path))

    @property
    def test_paths(self) -> List[str]:
        """ List of paths to all CSVs in test directory. """
        test_file_paths = [os.path.join(self.test_path, f) for f in self.test_filenames]
        return test_file_paths

    def meta_path(self, csv_name: str) -> str:
        """ Path to csv_name in metadata directory. """
        if not self.has_meta():
            raise DatasetException("Dataset has no metadata!")
        return os.path.join(self.path, 'meta/', csv_name)

    def index(self, df: pd.DataFrame, idx=None) -> pd.DataFrame:
        if idx is None:
            idx = self.index_colname
        df.index = df[idx]
        return df


class MultiDatasetException(Exception):
    pass


class MultiDataset(object):
    """
    Provides wrapper for reading multiple dataset folders as a single Dataset.
    It also supports using a cleaning strategy - a custom function that will transfom train.X and test dataframes in the same way.
    For now this is only intended to serve a train and predict interface,
    we can easily extend it to implement full Dataset interface but this is not necessary right now.
    """
    def __init__(self, paths: List[str], y_colname: str=None, index_colname='object_id', cleaning_strategy=lambda x:x):
        self.datasets = [Dataset(path, y_colname, index_colname) for path in paths]
        self._clean = cleaning_strategy

    @property
    def train(self) -> Tuple[pd.DataFrame, pd.Series]:
        base_X, base_y = self.datasets[0].train
        other_Xs = [self._valid_train_X(other_ds, base_y) for other_ds in self.datasets[1:]]
        joined_X = self._safe_join(base_X, other_Xs)
        return self._clean(joined_X), base_y
    
    @property
    def iter_test(self):
        if not self._children_have_same_test_split():
            raise MultiDatasetException("Internal datasets have differently aligned test CSVs")
        for test_dfs in zip(*[ds.iter_test for ds in self.datasets]):
            joined = self._safe_join(test_dfs[0], test_dfs[1:])
            yield self._clean(joined)

    def _children_have_same_test_split(self):
        first_child_filenames = set(self.datasets[0].test_filenames)
        for other_ds in self.datasets[1:]:
            if not set(other_ds.test_filenames) == first_child_filenames:
                return False
        return True

    def _valid_train_X(self, ds: Dataset, base_y: pd.Series) -> pd.DataFrame:
        X, y = ds.train
        if not (y == base_y).all():
            raise MultiDatasetException("Internal datasets have different target series")
        return X

    def _safe_join(self, base_X: pd.DataFrame, other_Xs: List[pd.DataFrame]) -> pd.DataFrame:
        for idx, X in enumerate(other_Xs):
            X.drop(columns=[col for col in set(base_X.columns) & set(X.columns)], inplace=True)
            base_X = base_X.join(X)
        return base_X


def build_dataset_structure(path: str, with_meta=False):
    """
    Creates an empty dataset directory structure in provided path and
    returns a Dataset instance with this path.
    """
    os.makedirs(os.path.join(path, 'test/'))
    if with_meta:
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
            save_batch(save_df, output_dir)
            progressbar.update(len(save_df))
        # save last reminder
        if len(reminder_df) > 0:
            save_batch(reminder_df, output_dir)
            progressbar.update(len(reminder_df))


def save_batch(batch: pd.DataFrame, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_filename = _name_for_batch(batch)
    batch.to_csv(os.path.join(output_dir, save_filename), index=False)


def _name_for_batch(batch: pd.DataFrame, pad_to_length=12) -> str:
    min_id_str = str(batch['object_id'].min()).zfill(pad_to_length)
    max_id_str = str(batch['object_id'].max()).zfill(pad_to_length)
    return f"test-batch-{min_id_str}-{max_id_str}.csv"
