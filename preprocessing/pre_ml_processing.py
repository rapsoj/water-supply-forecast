import datetime as dt
import os

import numpy as np
import pandas as pd

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
    start_date = df.date.min()
    end_date = df.date.max()
    feat_dates = pd.date_range(start=start_date, end=end_date, freq=f'{N_DAYS_DELTA}D')
    # todo we're throwing away data that we have here, can/should we use it eg for training?
    feat_dates = feat_dates[(feat_dates.month < 4) | (feat_dates.month > 9)].to_series()
    site_feat_cols = set(df.columns) - ({'site_id', 'date', 'forecast_year', 'station'} | set(date_cols))

    # average over data from different stations in the same day, todo - deal with this properly by using lat/lon data or something groovier
    site_id = df.name
    df = df.groupby('date')[list(site_feat_cols)].agg(lambda x: x.dropna().mean()).reset_index()

    # re-add global data into specific df todo use new data, currently this explods because of bad joins in the existing dataset
    df = df.merge(mjo_data, on='date', how='outer') \
        .merge(nino_data.drop_duplicates(), on='date', how='outer') \
        .merge(oni_data.drop_duplicates(), on='date', how='outer') \
        .merge(misc_data.drop_duplicates(), on='date', how='outer') \
        .sort_values(by='date') \
        .reset_index(drop=True)

    # Get associated dates
    orig_dates_vals = (df.date - start_date).dt.days
    feat_dates_vals = (feat_dates - start_date).dt.days

    # split into columns you interpolate and those you take the closest preceding value
    interp_cols = global_oni_cols + ['volume']
    interp_df = df[interp_cols].apply(lambda x: np.interp(feat_dates_vals, orig_dates_vals[~x.isna()],
                                                          x.dropna())).reset_index(drop=True)
    other_cols = list(set(df.columns) - set(interp_cols) - {'date'})

    def fill_preceding_val(column: pd.Series) -> pd.Series:
        col_inds = np.maximum(np.searchsorted(orig_dates_vals[~column.isna()], feat_dates_vals) - 1, 0)
        return column.dropna().iloc[col_inds].reset_index(drop=True)

    other_cols_df = df[other_cols].apply(fill_preceding_val).reset_index(drop=True)

    site_df = interp_df.join(other_cols_df)
    site_df['date'] = feat_dates.reset_index(drop=True)
    site_df['forecast_year'] = feat_dates.apply(lambda x: x.year + (x.month > 4)).reset_index(
        drop=True)  # set value as +1 for all

    # todo make sure this is after the train/test split, don't want leakage
    assert not site_df.isna().any().any(), 'Error - we have nans!'

    site_df['site_id'] = site_id
    return site_df


def ml_preprocess_data(data: pd.DataFrame, output_file_path: str = 'ml_processed_data.csv',
                       load_from_cache: bool = False) -> tuple:
    if load_from_cache and os.path.exists(output_file_path):
        return pd.read_csv(output_file_path)

    data = data.copy()

    # Read data
    other_cols = ["oniSEAS_AMJ", "oniSEAS_ASO", "oniSEAS_DJF",
                  "oniSEAS_FMA", "oniSEAS_JAS", "oniSEAS_JFM", "oniSEAS_JJA", "oniSEAS_MAM",
                  "oniSEAS_MJJ", "oniSEAS_NDJ", "oniSEAS_OND", "oniSEAS_SON", "site_id_american_river_folsom_lake",
                  "site_id_animas_r_at_durango", "site_id_boise_r_nr_boise", "site_id_boysen_reservoir_inflow",
                  "site_id_colville_r_at_kettle_falls",
                  "site_id_detroit_lake_inflow", "site_id_dillon_reservoir_inflow",
                  "site_id_fontenelle_reservoir_inflow",
                  "site_id_green_r_bl_howard_a_hanson_dam", "site_id_hungry_horse_reservoir_inflow",
                  "site_id_libby_reservoir_inflow",
                  "site_id_merced_river_yosemite_at_pohono_bridge", "site_id_missouri_r_at_toston",
                  "site_id_owyhee_r_bl_owyhee_dam",
                  "site_id_pecos_r_nr_pecos", "site_id_pueblo_reservoir_inflow", "site_id_ruedi_reservoir_inflow",
                  "site_id_san_joaquin_river_millerton_reservoir", "site_id_skagit_ross_reservoir",
                  "site_id_snake_r_nr_heise",
                  "site_id_stehekin_r_at_stehekin", "site_id_sweetwater_r_nr_alcova",
                  "site_id_taylor_park_reservoir_inflow",
                  "site_id_virgin_r_at_virtin", "site_id_weber_r_nr_oakley", "site_id_yampa_r_nr_maybell"]
    col2dtype = {"mjo20E": float, "mjo70E": float, "mjo80E": float, "mjo100E": float, "mjo120E": float,
                 "mjo140E": float, "mjo160E": float, "mjo120W": float, "mjo40W": float, "mjo10W": float,
                 "year": int, "month": int, "day": float, "ninoNINO1+2": float, "ninoANOM": float,
                 "ninoNINO3": float, "ninoANOM.1": float, "ninoNINO4": float, "ninoANOM.2": float,
                 "ninoNINO3.4": float, "ninoANOM.3": float, "oniTOTAL": float,
                 "oniANOM": float, "pdo": float, "pna": float, "soi_anom": float, "soi_sd": float,
                 "forecast_year": float, "volume": float, "mean_gws_inst": float, "mean_rtzsm_inst": float,
                 "mean_sfsm_inst": float, "station": float, "PREC_DAILY": float, "WTEQ_DAILY": float,
                 "TAVG_DAILY": float,
                 "TMAX_DAILY": float, "TMIN_DAILY": float, "SNWD_DAILY": float}

    for col in other_cols:
        col2dtype[col] = bool

    # monthly measurements are approx at the middle
    data.day[data.day == -1] = 15

    # Reformulate ONIXXX from one hot encoding to date
    oni_temporal_cols = [col for col in data.columns if 'oniSEAS' in col]
    month_conversion_dictionary = {
        'oniSEAS_AMJ': 5,
        'oniSEAS_ASO': 9,
        'oniSEAS_DJF': 1,
        'oniSEAS_FMA': 3,
        'oniSEAS_JAS': 8,
        'oniSEAS_JFM': 2,
        'oniSEAS_JJA': 7,
        'oniSEAS_MAM': 4,
        'oniSEAS_MJJ': 6,
        'oniSEAS_NDJ': 12,
        'oniSEAS_OND': 11,
        'oniSEAS_SON': 10
    }
    data['oni_month'] = data[oni_temporal_cols] \
        .idxmax(axis='columns') \
        .apply(month_conversion_dictionary.get)

    # todo finish oni date conversion/check that it works properly

    data.month[data.month == -1] = data.oni_month[data.month == -1]

    # Create integer dates to work with
    data["date"] = pd.to_datetime(data[date_cols].applymap(int))
    data = data.sort_values('date')

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
    california_sites = ['american_river_folsom_lake',
                        'merced_river_yosemite_at_pohono_bridge',
                        'san_joaquin_river_millerton_reservoir']
    missing_snotel_site_mask = data.site_id.isin(california_sites)
    processed_data = data[~missing_snotel_site_mask].groupby('site_id').apply(process_features, mjo_data=mjo_data,
                                                                              nino_data=nino_data, oni_data=oni_data,
                                                                              misc_data=misc_data) \
        .reset_index(drop=True)

    processed_data.to_csv(output_file_path, index=False)

    processed_ground_truth = pd.DataFrame()
    processed_ground_truth['gt'] = data.volume.dropna()

    processed_ground_truth['date'] = data[~data.volume.isna()].date
    processed_ground_truth['site_id'] = data.site_id
    processed_ground_truth['forecast_year'] = data.date.dt.year
    return processed_data, processed_ground_truth
