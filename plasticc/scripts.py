from argparse import ArgumentParser

import click
import pandas as pd

from plasticc.dataset import batch_data as batch_data_func


@click.command()
@click.option('--input-csv', help='Input csv path containing time series', required=True)
@click.option('--output-dir', help='Output directory path', required=True)
@click.option('--batch-size', type=int, default=150000, help='Batch size in rows')
@click.option('--input-rows', type=int, default=453653105, help='Rows in input csv')
@click.option('--meta-csv', type=str, default=None, help='File with metadata to merge with input csv')
def batch_data(input_csv, output_dir, batch_size, input_rows, meta_csv):
    reader = pd.read_csv(input_csv, chunksize=batch_size)
    meta_df = pd.read_csv(meta_csv) if meta_csv is not None else None
    batch_data_func(reader, meta_df=meta_df, output_dir=output_dir, lines=input_rows)
    print('Done.')
