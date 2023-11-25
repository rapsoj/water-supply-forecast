import pandas as pd
import os
import numpy as np
import datetime as dt

path = os.getcwd()
# Set working directory to yours
# os.chdir("/Users/emilryd/programming/water-supply-forecast")

# Read data
other_cols = ["oniSEAS_AMJ", "oniSEAS_ASO", "oniSEAS_DJF",
              "oniSEAS_FMA", "oniSEAS_JAS", "oniSEAS_JFM", "oniSEAS_JJA", "oniSEAS_MAM",
              "oniSEAS_MJJ", "oniSEAS_NDJ", "oniSEAS_OND", "oniSEAS_SON", "site_id_american_river_folsom_lake",
              "site_id_animas_r_at_durango", "site_id_boise_r_nr_boise", "site_id_boysen_reservoir_inflow",
              "site_id_colville_r_at_kettle_falls",
              "site_id_detroit_lake_inflow", "site_id_dillon_reservoir_inflow", "site_id_fontenelle_reservoir_inflow",
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
             "mean_sfsm_inst": float, "station": float, "PREC_DAILY": float, "WTEQ_DAILY": float, "TAVG_DAILY": float,
             "TMAX_DAILY": float, "TMIN_DAILY": float, "SNWD_DAILY": float}

for col in other_cols:
    col2dtype[col] = bool
data = pd.read_csv(os.path.join("..", "02-data-cleaning", "transformed_vars.csv"),
                   dtype=col2dtype)

# Create integer dates to work with
data["date"] = pd.to_datetime(data[['year', 'month', 'day']].applymap(int))
data = data.sort_values('date')

# Get site ids
site_id_str = 'site_id_'
data['site_id'] = data[[col for col in data.columns if 'site_id' in col]] \
    .idxmax(axis='columns') \
    .apply(lambda x: x[x.find(site_id_str) + len(site_id_str):])
data = data.drop(columns=[col for col in data.columns if site_id_str in col])

# Prediction months and dates
prediction_dates = [dt.datetime(year, month, day) for year in data.year.unique() for month in range(4, 8) for day in
                    range(1, 30, 7)]


# Create training set for a site_id
def process_features(df: pd.DataFrame):
    print()


data.groupby('site_id').apply(process_features)
