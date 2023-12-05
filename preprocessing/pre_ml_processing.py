import datetime as dt
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

path = os.getcwd()

global_mjo_cols = ['mjo20E', 'mjo70E', 'mjo80E', 'mjo100E', 'mjo120E', 'mjo140E', 'mjo160E', 'mjo120W', 'mjo40W',
                   'mjo10W']
global_oni_cols = ['oniTOTAL', 'oniANOM']
global_nino_cols = ['ninoNINO1+2', 'ninoANOM', 'ninoNINO3', 'ninoANOM.1', 'ninoNINO4', 'ninoANOM.2',
                    'ninoNINO3.4', 'ninoANOM.3']
global_misc_cols = ['pdo', 'pna', 'soi_anom', 'soi_sd']

shared_cols = ['date']

date_cols = ['year', 'month', 'day']


def process_features(df: pd.DataFrame, mjo_data: pd.DataFrame, nino_data: pd.DataFrame, oni_data: pd.DataFrame,
                     misc_data: pd.DataFrame, N_DAYS_DELTA: int = 7) -> pd.DataFrame:
    # Generate a df with rows for every prediction date, then gather data accordingly up until that point
    start_date1 = pd.to_datetime(f"{df.date.dt.year.min()}0101", format="%Y%m%d")
    # start_time = time.strptime(start_date1)
    end_date1 = pd.to_datetime(f"{df.date.dt.year.max()}0701", format="%Y%m%d")

    end_date = df.date.max()
    feat_dates1 = pd.date_range(start=start_date1, end=end_date1, freq=f'{1}MS')
    feat_dates2 = feat_dates1.shift(7, freq="D")
    feat_dates3 = feat_dates1.shift(14, freq="D")
    feat_dates4 = feat_dates1.shift(21, freq="D")
    feat_dates = pd.Series(np.sort(
        np.concatenate((np.array(feat_dates1), np.array(feat_dates2), np.array(feat_dates3), np.array(feat_dates4)))))

    site_feat_cols = set(df.columns) - ({'site_id', 'date', 'forecast_year', 'station'} | set(date_cols))

    # average over data from different stations in the same day, todo - deal with this properly by using lat/lon data or something groovier
    site_id = df.name
    df = df.groupby('date')[list(site_feat_cols)].agg(lambda x: x.dropna().mean()).reset_index()

    # drop irrelevant columns, especially relevant for california data that's missing some features
    df = df.drop(columns=df.columns[df.isna().all()])

    # re-add global data into specific df
    df = df.merge(mjo_data, on='date', how='outer') \
        .merge(nino_data.drop_duplicates(), on='date', how='outer') \
        .merge(oni_data.drop_duplicates(), on='date', how='outer') \
        .merge(misc_data.drop_duplicates(), on='date', how='outer') \
        .sort_values(by='date') \
        .reset_index(drop=True)

    # Get associated dates
    orig_dates_vals = (df.date - start_date1).dt.days
    feat_dates_vals = (feat_dates - start_date1).dt.days

    # split into columns you interpolate and those you take the closest preceding value
    interp_cols = list(set(global_oni_cols + ['volume']) & set(df.columns))
    interp_df = df[interp_cols].apply(lambda x: np.interp(feat_dates_vals, orig_dates_vals[~x.isna()],
                                                          x.dropna())).reset_index(drop=True)
    other_cols = list(set(df.columns) - set(interp_cols) - {'date'})

    def fill_preceding_val(column: pd.Series) -> pd.Series:
        col_inds = np.maximum(np.searchsorted(orig_dates_vals[~column.isna()], feat_dates_vals) - 1, 0)
        return column.dropna().iloc[col_inds].reset_index(drop=True)

    other_cols_df = df[other_cols].apply(fill_preceding_val).reset_index(drop=True)

    site_df = interp_df.join(other_cols_df)
    site_df['date'] = feat_dates.reset_index(drop=True)
    site_df['forecast_year'] = feat_dates.apply(lambda x: x.year + (x.month >= 10)).reset_index(
        drop=True)  # set value as +1 for all

    # todo make sure this is after the train/test split, don't want leakage
    assert not site_df.isna().any().any(), 'Error - we have nans!'

    site_df['site_id'] = site_id
    return site_df


def ml_preprocess_data(data: pd.DataFrame, output_file_path: str = 'ml_processed_data.csv',
                       load_from_cache: bool = False) -> tuple:
    if load_from_cache and os.path.exists(output_file_path):
        return pd.read_csv(output_file_path, parse_dates=['date'])

    data = data.copy()

    # monthly measurements are approx at the middle
    data.day[data.day == -1] = 15

    # Create dates to work with
    data["date"] = pd.to_datetime(data[date_cols].map(int))
    data = data.sort_values('date')

    ini_data = data
    # Get site ids
    site_id_str = 'site_id_'
    site_id_cols = [col for col in data.columns if 'site_id' in col]

    data['site_id'] = data[site_id_cols] \
        .idxmax(axis='columns') \
        .apply(lambda x: x[x.find(site_id_str) + len(site_id_str):])
    data = data.drop(columns=site_id_cols)

    # todo make sure you re-incorporate all of these+interpolate them properly
    mjo_data = data[global_mjo_cols + shared_cols].dropna()
    oni_data = data[global_oni_cols + shared_cols].dropna()

    nino_data = data[global_nino_cols + shared_cols].dropna()
    misc_data = data[global_misc_cols + shared_cols].dropna()
    # Keeping only SNOTEL data
    data = data.drop(columns=global_mjo_cols + global_nino_cols + global_oni_cols + global_misc_cols)

    # Removing sites with no snotel data
    # todo process this data separately
    processed_data = data.groupby('site_id').apply(process_features, mjo_data=mjo_data, nino_data=nino_data,
                                                   oni_data=oni_data, misc_data=misc_data) \
        .reset_index(drop=True)

    processed_data.to_csv(output_file_path, index=False)

    return processed_data

