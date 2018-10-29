import os

import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm


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
