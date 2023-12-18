import os
from functools import reduce

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from consts import OCTOBER, JULY, MID_MONTH_DAY

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
                     misc_data: pd.DataFrame) -> pd.DataFrame:
    # Generate a df with rows for every prediction date, then gather data accordingly up until that point
    start_date1 = pd.to_datetime(f"{df.date.dt.year.min()}0101", format="%Y%m%d")
    end_date1 = pd.to_datetime(f"{df.date.dt.year.max()}0701", format="%Y%m%d")

    feat_dates1 = pd.date_range(start=start_date1, end=end_date1, freq=f'{1}MS')
    feat_dates2 = feat_dates1.shift(7, freq="D")
    feat_dates3 = feat_dates1.shift(14, freq="D")
    feat_dates4 = feat_dates1.shift(21, freq="D")
    feat_dates = pd.Series(np.sort(
        np.concatenate((np.array(feat_dates1), np.array(feat_dates2), np.array(feat_dates3), np.array(feat_dates4)))))

    site_id = df.name

    # break every station into its separate columns (while keeping the df which has station=NaN separate,
    # merging it back later to make sure it doesn't disappear)
    # todo - deal with stations more properly by using lat/lon data or something groovier
    def expand_columns(dataf: pd.DataFrame, expansion_cols):
        for expansion_col in expansion_cols:
            rest_df = dataf[dataf[expansion_col].isna()]
            expansion_df = dataf[~dataf[expansion_col].isna()]
            if not dataf[expansion_col].isna().all():

                if expansion_col == 'station':
                    renaming_cols = [col for col in dataf.columns if 'DAILY' in col] + [expansion_col]
                elif expansion_col == 'LEAD':
                    renaming_cols = [col for col in dataf.columns if '_prec' in col or '_temp' in col] + [expansion_col]

                expansion_dfs = [group for _, group in expansion_df.groupby(expansion_col)]

                new_expansion_dfs = [station_dataf.rename(
                    columns={col: (col + str(station_dataf[expansion_col].unique()[0])) for col in renaming_cols}) for
                    station_dataf in expansion_dfs]
                if expansion_col == 'station':
                    unshared_cols = [col for col in dataf.columns if 'DAILY' in col] + [expansion_col]
                elif expansion_col == 'LEAD':
                    unshared_cols = [col for col in dataf.columns if '_prec' in col or '_temp' in col] + [expansion_col]

                shared_cols = list(set(dataf.columns) - set(unshared_cols))
                merging_cols = ['year', 'month', 'day'] + \
                               list(set(shared_cols) - set(['year', 'month', 'day'] + [expansion_col]))

                expansion_df = reduce(lambda left, right: pd.merge(left, right, on=merging_cols,
                                                                   how='outer'), new_expansion_dfs)

            dataf = pd.concat((expansion_df, rest_df))
        return dataf

    # df = expand_columns(df, expansion_cols=['station', 'LEAD'])
    # drop irrelevant columns, especially relevant for california data that's missing some features
    df = df.drop(columns=df.columns[df.isna().all()])
    # drop station name columns
    station_cols = [col for col in df.columns if 'station' in col]
    df = df.drop(columns=station_cols)
    site_feat_cols = set(df.columns) - ({'site_id', 'date', 'forecast_year', 'station'} | set(date_cols))

    # todo - do not average over all cpc forecasts with different leads on the same date, deal with it in a smarter/more information preserving manner
    # df = df.groupby('date')[list(site_feat_cols)].agg(lambda x: x.dropna().mean()).reset_index()

    # todo interpolate variables that only stretch a certain extent back in time such that they take the average value for everything after (i.e. 0s or a site-wise average), e.g. for CPC forecasts

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
    site_df['forecast_year'] = feat_dates.apply(lambda x: x.year + (x.month >= OCTOBER)).reset_index(drop=True)

    # todo make sure this is after the train/test split, don't want leakage
    assert not site_df.isna().any().any(), 'Error - we have nans!'

    site_df['site_id'] = site_id
    print(f'Finishing processing features of {site_id}')
    return site_df


def ml_preprocess_data(data: pd.DataFrame, output_file_path: str = 'ml_processed_data.csv',
                       load_from_cache: bool = False) -> tuple:
    if load_from_cache and os.path.exists(output_file_path):
        return pd.read_csv(output_file_path, parse_dates=['date'])

    data = data.copy()

    # cannot use data outside of water year
    data = data[(data.date.dt.month <= JULY) | (data.date.dt.month > OCTOBER)] \
        .reset_index(drop=True)

    # monthly measurements are approx at the middle
    data.day[data.day == -1] = MID_MONTH_DAY

    # Create dates to work with
    data["date"] = pd.to_datetime(data[date_cols].applymap(int))
    data = data.sort_values('date')
    data = data.drop(columns=date_cols)

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

    processed_data = data.groupby('site_id').apply(process_features, mjo_data=mjo_data, nino_data=nino_data,
                                                   oni_data=oni_data, misc_data=misc_data) \
        .reset_index(drop=True)

    # adding temporal feature
    def calculate_first_of_oct(date):
        year = date.year if date.month >= OCTOBER else date.year - 1
        return pd.Timestamp(year=year, month=OCTOBER, day=1)

    processed_data['time'] = processed_data.date.apply(lambda x: (x - calculate_first_of_oct(x)).days)

    scaler = StandardScaler()
    processed_data.time = scaler.fit_transform(processed_data[['time']])

    processed_data.to_csv(output_file_path, index=False)

    return processed_data
