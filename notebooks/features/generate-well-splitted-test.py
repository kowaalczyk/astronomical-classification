""" Helper script for generating features so that we can examine terminal output after reconnecting. """

import pandas as pd
import numpy as np
from tqdm import tqdm

import time
import os
from datetime import datetime


chunk_size_one_100 = 453653105 // 100 + 1  # should take 100x (time after 1st iteration)


meta_path='../../data/raw/test_set_metadata.csv'
test_path='../../data/raw/test_set.csv'
output_path='../../data/raw/test-splits/'
id_colname='object_id'
chunks=chunk_size_one_100  # alternatively: 5000000


if __name__ == '__main__':
    os.makedirs(output_path)    

    remain_df = None
    for i_c, df in tqdm(enumerate(pd.read_csv(test_path, chunksize=chunks, iterator=True)), total=100):
        unique_ids = np.unique(df[id_colname])
        new_remain_df = df.loc[df[id_colname] == unique_ids[-1]].copy()
        if remain_df is None:
            df = df.loc[df[id_colname].isin(unique_ids[:-1])]
        else:
            df = pd.concat([remain_df, df.loc[df[id_colname].isin(unique_ids[:-1])]], axis=0)
        remain_df = new_remain_df
        
        chunk_save_path = os.path.join(output_path, f'test-raw-{df[id_colname].iloc[0]}-{df[id_colname].iloc[-1]}.csv')
        df.to_csv(chunk_save_path, index=False)
    remain_save_path = os.path.join(output_path, f'test-raw-{remain_df[id_colname].iloc[0]}-{remain_df[id_colname].iloc[-1]}.csv')
    remain_df.to_csv(chunk_save_path, index=False)
    print("Done!")
