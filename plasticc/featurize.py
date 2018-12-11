import gc

import numpy as np
import pandas as pd

from numba import jit
from tsfresh.feature_extraction import extract_features

gc.enable()
np.warnings.filterwarnings('ignore')


@jit
def haversine_plus(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) from
    # https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    # Convert decimal degrees to Radians:
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    # Implementing Haversine Formula:
    dlon = np.subtract(lon2, lon1)
    dlat = np.subtract(lat2, lat1)

    a = np.add(np.power(np.sin(np.divide(dlat, 2)), 2),
               np.multiply(
                   np.cos(lat1),
                   np.multiply(np.cos(lat2),
                               np.power(np.sin(
                                   np.divide(dlon, 2)), 2))))

    haversine = np.multiply(2, np.arcsin(np.sqrt(a)))
    return {
        'haversine': haversine,
        'latlon1': np.subtract(np.multiply(lon1, lat1),
                               np.multiply(lon2, lat2)),
    }


def process_meta(filename):
    meta_df = pd.read_csv(filename)
    meta_dict = dict()
    # distance
    meta_dict.update(haversine_plus(meta_df['ra'].values, meta_df['decl'].values,
                                    meta_df['gal_l'].values, meta_df['gal_b'].values))

    meta_dict['hostgal_photoz_certain'] = np.multiply(
        meta_df['hostgal_photoz'].values,
        np.exp(meta_df['hostgal_photoz_err'].values))

    meta_df = pd.concat([meta_df, pd.DataFrame(meta_dict, index=meta_df.index)], axis=1)
    return meta_df


@jit
def process_flux(df):
    flux_ratio_sq = np.power(df['flux'].values / df['flux_err'].values, 2.0)

    df_flux = pd.DataFrame({
        'flux_ratio_sq': flux_ratio_sq,
        'flux_by_flux_ratio_sq': df['flux'].values * flux_ratio_sq},
        index=df.index
    )
    return pd.concat([df, df_flux], axis=1)


@jit
def process_flux_agg(df):
    flux_w_mean = df['flux_by_flux_ratio_sq_sum'].values / df['flux_ratio_sq_sum'].values
    flux_diff = df['flux_max'].values - df['flux_min'].values

    df_flux_agg = pd.DataFrame({
        'flux_w_mean': flux_w_mean,
        'flux_diff1': flux_diff,
        'flux_diff2': flux_diff / df['flux_mean'].values,
        'flux_diff3': flux_diff / flux_w_mean,
        }, index=df.index)

    return pd.concat([df, df_flux_agg], axis=1)


def featurize(df, df_meta, aggs, fcp, n_jobs=4):
    """
    Extracting Features from train set
    Features from olivier's kernel
    very smart and powerful feature that is generously given here
    https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
    per passband features with tsfresh library. fft features added
    to capture periodicity
    https://www.kaggle.com/c/PLAsTiCC-2018/discussion/70346#415506
    """
    df = process_flux(df)

    agg_df = df.groupby('object_id').agg(aggs)
    agg_df.columns = ['{}_{}'.format(k, agg)
                      for k in aggs.keys() for agg in aggs[k]]
    agg_df = process_flux_agg(agg_df)  # new feature to play with tsfresh

    # Add more features with tsfresh:
    agg_df_ts_flux_passband = extract_features(
        df,
        column_id='object_id',
        column_sort='mjd',
        column_kind='passband',
        column_value='flux',
        default_fc_parameters=fcp['flux_passband'], 
        n_jobs=n_jobs
    )
    agg_df_ts_flux = extract_features(
        df,
        column_id='object_id',
        column_value='flux',
        default_fc_parameters=fcp['flux'],
        n_jobs=n_jobs
    )
    agg_df_ts_flux_by_flux_ratio_sq = extract_features(
        df,
        column_id='object_id',
        column_value='flux_by_flux_ratio_sq',
        default_fc_parameters=fcp['flux_by_flux_ratio_sq'],
        n_jobs=n_jobs
    )
    # Add smart feature that is suggested here
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
    # dt[detected==1, mjd_diff:=max(mjd)-min(mjd), by=object_id]
    df_det = df[df['detected'] == 1].copy()
    agg_df_mjd = extract_features(
        df_det,
        column_id='object_id',
        column_value='mjd',
        default_fc_parameters=fcp['mjd'],
        n_jobs=n_jobs
    )
    agg_df_mjd['mjd_diff_det'] = agg_df_mjd['mjd__maximum'].values - agg_df_mjd['mjd__minimum'].values
    del agg_df_mjd['mjd__maximum'], agg_df_mjd['mjd__minimum']

    agg_df_ts_flux_passband.index.rename('object_id', inplace=True)
    agg_df_ts_flux.index.rename('object_id', inplace=True)
    agg_df_ts_flux_by_flux_ratio_sq.index.rename('object_id', inplace=True)
    agg_df_mjd.index.rename('object_id', inplace=True)
    agg_df_ts = pd.concat(
        [agg_df, agg_df_ts_flux_passband,
         agg_df_ts_flux,
         agg_df_ts_flux_by_flux_ratio_sq,
         agg_df_mjd],
        axis=1
    ).reset_index()
    result = agg_df_ts.merge(right=df_meta, how='left', on='object_id')
    result.fillna(0, inplace=True)
    return result
