import gc
from typing import Tuple
from functools import partial
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.signal import lombscargle, find_peaks
from scipy.optimize import curve_fit
from sklearn.preprocessing import minmax_scale
from tsfresh.feature_extraction import extract_features

from numba import jit

gc.enable()
np.warnings.filterwarnings('ignore')

# features for metadata, from Kernel

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

# main metadata processing function

def process_meta(filename):
    meta_df = pd.read_csv(filename)
    meta_dict = dict()
    # distance
    meta_dict.update(haversine_plus(meta_df['ra'].values, meta_df['decl'].values,
                                    meta_df['gal_l'].values, meta_df['gal_b'].values))

    meta_dict['hostgal_photoz_certain'] = np.multiply(
        meta_df['hostgal_photoz'].values,
        np.exp(meta_df['hostgal_photoz_err'].values))
    
    meta_dict['z_distance_approx'] = (meta_df['hostgal_photoz'] + 1)**2

    meta_df = pd.concat([meta_df, pd.DataFrame(meta_dict, index=meta_df.index)], axis=1)
    return meta_df

# features for time series, from the Kernel

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

# custom normalized series groupby features

@jit
def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

lombscargle_freqs = np.linspace(0.01, 4, 400)

@jit
def process_passband_features(x: np.array, y: np.array) -> Tuple[np.array, np.array]:
    # raw peaks
    peaks, _ = find_peaks(y)
    if len(peaks) > 0:
        peak_values = y.take(peaks)
        peak_time = x.take(peaks)
        peak_features = np.array([
            peak_time.min(),
            peak_time.max(),
            len(peaks),
            peak_time.mean(),
            peak_values.min(),
            peak_values.max(),
            peak_values.mean(),
            peak_values.std(),
        ], dtype=np.float64)
    else:
        peak_features = np.zeros(8, dtype=np.float64)
    peak_names =  np.array([
        'peak_time_min',
        'peak_time_max',
        'peaks_count',
        'peak_time_mean',
        'peak_values_min',
        'peak_values_max',
        'peak_values_mean',
        'peak_values_std',
    ], dtype=str)
    # period
    periodogram = lombscargle(x, y, freqs=lombscargle_freqs)
    peaks, _ = find_peaks(periodogram)
    if len(peaks) > 0:
        peak_values = periodogram.take(peaks)
#         peak_time = minmax_scale(peaks, feature_range=(-1,1))
        period_features = np.array([
            peaks.min(),
            peaks.max(),
            len(peaks),
            peaks.mean(),
            peak_values.min(),
            peak_values.max(),
            peak_values.mean(),
            peak_values.std(),
        ], dtype=np.float64)
    else:
        period_features = np.zeros(8, dtype=np.float64)
    period_names = np.array([
        'period_peak_time_min',
        'period_peak_time_max',
        'period_peaks_count',
        'period_peak_time_mean',
        'period_peak_values_min',
        'period_peak_values_max',
        'period_peak_values_mean',
        'period_peak_values_std',
    ], dtype=str)
    # linear
    lin = linregress(x, y)
    linear_features = np.array(lin, dtype=np.float64)
    linear_names = np.array([
        'slope',
        'intercept',
        'r-value',
        'p-value',
        'stderr'
    ], dtype=str)
    # poly
    poly, residuals, rank, singular_values, rcond = np.polyfit(x, y, deg=5, full=True, cov=False)
    if len(residuals) > 0:
        residuals_features = np.array([
            residuals.min(),
            residuals.max(),
            residuals.mean(),
        ], dtype=np.float64)
    else:
        residuals_features = np.array([0,0,0], dtype=np.float64)
    residuals_names = np.array([
        'poly_residuals_min',
        'poly_residuals_max',
        'poly_residuals_mean',
    ], dtype=str)
    poly_features = np.concatenate([poly[:-1], residuals_features])
    poly_names = np.concatenate([
        np.array(['poly_coeff_'+str(i) for i in range(len(poly)-1)], dtype=str),
        residuals_names
    ])
    # gaussian
#     try:
#         popt, pcov = curve_fit(gauss_function, x, y, p0 = [1., 0., 1.], method='trf')
#     except:
#         popt, pcov = np.zeros(3, dtype=np.float64), np.zeros((3,3), dtype=np.float64)
#     if np.isinf(pcov).any():
#         gauss_std_err = np.zeros(3, dtype=np.float64)
#     else:
#         gauss_cov_diag = np.diag(pcov)
#         gauss_std_err = np.sqrt(gauss_cov_diag)
#     gaussian_features = np.concatenate([popt, gauss_std_err])
#     gaussian_names = np.array([
#         'gaussian_a',
#         'gaussian_x0',
#         'gaussian_sigma',
#         'gaussian_a_err',
#         'gaussian_x0_err',
#         'gaussian_sigma_err',
#     ], dtype=str)
    # combine all lists
    features = np.concatenate([peak_features, period_features, linear_features, poly_features])  # , gaussian_features])
    feature_names = np.concatenate([peak_names, period_names, linear_names, poly_names])  # , gaussian_names])
    return features, feature_names

def process_group(group_data_: Tuple[int, pd.DataFrame], passband_colname: str='passband_group') -> pd.Series:
    group_idx_, group_ = group_data_
    threshold = group_['flux_err'].mean() + 2.5*group_['flux_err'].std()
    filtered_group_ = group_[group_['flux_err'] <= threshold]
    agg_features = pd.Series()
    for passband_group_idx_, passband_group_ in group_.groupby(passband_colname):
        x, y = passband_group_['scaled_mjd'].values, passband_group_['flux'].values
        pb_features, pb_feature_names = process_passband_features(x,y)
        agg_features = agg_features.append(
            pd.Series(pb_features, index=pb_feature_names)\
            .add_prefix(f'pbgroup_{passband_group_idx_}_')
        )
    return agg_features.rename(group_idx_)

def calculate_series_features(preprocessed_df: pd.DataFrame, passband_colname: str='passband_group') -> pd.DataFrame:
    gbo = preprocessed_df.groupby('object_id')
    group_features_series = [process_group(g) for g in gbo]
    series_features_df = pd.concat(group_features_series, axis=1)
    return series_features_df.transpose()  # will be joined with other features at the end
    

# preprocessing for custom functions on time series

@jit(["int32(int32, float32)", "int64(int64, float64)"], nopython=True)
def fix_passband(raw_passband: int, redshift: float):
    """ 
    Moving from raw passband r to passband p, if: 
    redshift offset x high confidence min freq of r <= min freq of p, 
    redshift offset x and high confidence max freq of r >= max freq of p
    """
    freq_mean = np.array([350.0, 500.0, 600.0, 750.0, 875.0, 1000.0])
    freq_min = np.array([300.0, 400.0, 500.0, 650.0, 800.0, 950.0])
    freq_max = np.array([400.0, 600.0, 700.0, 850.0, 950.0, 1050.0])
    mean_with_offset = (1+redshift) * freq_mean[raw_passband]
    for i in range(6):
        if mean_with_offset > freq_min[i] and mean_with_offset < freq_max[i]:
            return i
    return np.nan

@jit(["int32[:](float32[:,:])", "int64[:](float64[:,:])"])
def fix_passband_vct(passband_and_redshift_):
    result = np.zeros(passband_and_redshift_.shape[0]).astype(np.int)
    for i in range(passband_and_redshift_.shape[0]):
        result[i] = fix_passband(passband_and_redshift_[i][0].astype(np.int), passband_and_redshift_[i][1])
    return result

def calculate_fixed_passband_and_scaled_flux(series_df_: pd.DataFrame, redshift_series: pd.Series) -> pd.Series:
    series_df = series_df_.copy()
    series_df['redshift'] = series_df_['object_id'].map(redshift_series).values
    series_df['fixed_passband'] = fix_passband_vct(series_df[['passband', 'redshift']].values).astype(np.int)
    scaling_factor = (series_df['redshift'] + 1)**2
    series_df['flux'] = series_df_['flux']*scaling_factor
    series_df['flux_err'] = series_df_['flux_err']*scaling_factor
    del series_df['redshift']
    return series_df


scale_time = lambda mjd_: pd.Series(minmax_scale(mjd_.values, feature_range=(-1,1)), index=mjd_.index)

def preprocess_series(series_df_: pd.DataFrame) -> pd.DataFrame:
    series_df = series_df_.copy()
    series_df['passband_group'] = series_df_['passband'] //2
    series_df['scaled_mjd'] = series_df_.groupby('object_id')['mjd'].apply(scale_time)
    return series_df  # not dropping raw passband and raw mjd because RAM is cheap

# main featurization functions combining all presently used features

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
    
    # start by scaling time and fixing passband alignment
    print("Preprocessing time...")
    preprocessed_df = preprocess_series(df)
    gc.collect()
    print("Fixing passband alignment...")
    redshift_series = pd.Series(df_meta['hostgal_photoz'].fillna(0).values, index=df_meta['object_id'])
    preprocessed_df_2 = calculate_fixed_passband_and_scaled_flux(preprocessed_df, redshift_series)
    
    # new, custom series features
    print("Generating custom features...")
    series_features_df = calculate_series_features(preprocessed_df)
    print("Generating custom features for fixed passbands...")
    series_features_df_fpb = calculate_series_features(preprocessed_df_2, passband_colname='fixed_passband').add_suffix('_fpb')
    print("Custom features generated.")
    total_nans = series_features_df.isna().any().sum()
    if total_nans > 0:
        print(f"WARNING: number of NaNs: {total_nans}")
        series_features_df.fillna(0, inplace=True)

    # features from the kernel
    
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
    agg_df_ts_flux_passband_fpb = extract_features(
        preprocessed_df_2,
        column_id='object_id',
        column_sort='mjd',
        column_kind='fixed_passband',
        column_value='flux',
        default_fc_parameters=fcp['flux_passband'], 
        n_jobs=n_jobs
    ).add_suffix('_preprocessed')
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
    agg_df_ts_flux_passband_fpb.index.rename('object_id', inplace=True)
    agg_df_ts_flux_by_flux_ratio_sq.index.rename('object_id', inplace=True)
    agg_df_mjd.index.rename('object_id', inplace=True)
    agg_df_ts = pd.concat([
        agg_df, 
        agg_df_ts_flux_passband,
        agg_df_ts_flux,
        agg_df_ts_flux_passband_fpb,
        agg_df_ts_flux_by_flux_ratio_sq,
        agg_df_mjd],
        axis=1
    ).reset_index()
    result = agg_df_ts.merge(right=df_meta, how='left', on='object_id')
    result.fillna(0, inplace=True)
    
    result = result.join(series_features_df, on='object_id')  # newly added series features
    result = result.join(series_features_df_fpb, on='object_id')  # newly added series features
    return result
