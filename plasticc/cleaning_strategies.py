""" 
Cleaning strategies to use with MultiDatasetClass.
See notebook "Dataset-Develpment" for example uses.
"""

import pandas as pd


def clean_kaggle_simple(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=[col for col in set(df.columns) if col.endswith('_meta')], inplace=True)
    df.drop(columns=['hostgal_specz'], inplace=True)  # for most samples in test set this is None
    df['distmod'] = df['distmod'].fillna(0)  # distance = 0 for objects in same galaxy
    return df
