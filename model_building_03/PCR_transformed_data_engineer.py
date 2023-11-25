import pandas as pd
import os
import numpy as np
import datetime as dt


path = os.getcwd()
# Set working directory to yours
# os.chdir("/Users/emilryd/programming/water-supply-forecast")

# Read data
list = ["oniSEAS_AMJ","oniSEAS_ASO","oniSEAS_DJF",
        "oniSEAS_FMA","oniSEAS_JAS","oniSEAS_JFM","oniSEAS_JJA","oniSEAS_MAM",
        "oniSEAS_MJJ","oniSEAS_NDJ","oniSEAS_OND","oniSEAS_SON","site_id_american_river_folsom_lake",
        "site_id_animas_r_at_durango","site_id_boise_r_nr_boise","site_id_boysen_reservoir_inflow","site_id_colville_r_at_kettle_falls",
        "site_id_detroit_lake_inflow","site_id_dillon_reservoir_inflow","site_id_fontenelle_reservoir_inflow",
        "site_id_green_r_bl_howard_a_hanson_dam","site_id_hungry_horse_reservoir_inflow","site_id_libby_reservoir_inflow",
        "site_id_merced_river_yosemite_at_pohono_bridge","site_id_missouri_r_at_toston","site_id_owyhee_r_bl_owyhee_dam",
        "site_id_pecos_r_nr_pecos","site_id_pueblo_reservoir_inflow","site_id_ruedi_reservoir_inflow",
        "site_id_san_joaquin_river_millerton_reservoir","site_id_skagit_ross_reservoir","site_id_snake_r_nr_heise",
        "site_id_stehekin_r_at_stehekin","site_id_sweetwater_r_nr_alcova","site_id_taylor_park_reservoir_inflow",
        "site_id_virgin_r_at_virtin","site_id_weber_r_nr_oakley","site_id_yampa_r_nr_maybell"]
dict = {"mjo20E": float, "mjo70E": float, "mjo80E": float, "mjo100E": float, "mjo120E": float,
                          "mjo140E": float, "mjo160E": float, "mjo120W": float, "mjo40W": float, "mjo10W": float,
                          "year": int, "month": int, "day": float, "ninoNINO1+2": float, "ninoANOM": float,
                          "ninoNINO3": float, "ninoANOM.1": float, "ninoNINO4": float, "ninoANOM.2": float,
                          "ninoNINO3.4": float, "ninoANOM.3": float, "oniTOTAL": float,
                          "oniANOM": float, "pdo": float, "pna": float, "soi_anom": float, "soi_sd": float,
                          "forecast_year": float, "volume": float, "mean_gws_inst": float, "mean_rtzsm_inst":float, 
                          "mean_sfsm_inst": float, "station": float, "PREC_DAILY": float, "WTEQ_DAILY": float, "TAVG_DAILY": float,
                          "TMAX_DAILY": float, "TMIN_DAILY": float, "SNWD_DAILY": float}

for i in list:
    dict[i] = bool
data = pd.read_csv(os.path.join("..", "02-data-cleaning", "merged_dataframe.csv"),
                   dtype=dict)

# Create integer dates to work with
data["dateint"] = data["year"]*1e4+data["month"]*1e2+data["day"]

# Get site ids
site_ids = np.unique(data["site_id"].to_numpy()[:300])

# Get years
years = np.unique(np.floor(data["year"].to_numpy() / 1e4))

# Prediction months and days
prediction_dates = np.zeros()

# Create training set for a site_id
for site_id in site_ids:

    # Iterate over every year:
    for year in years:
        # Now the fun begins
        # Iterate over every site you want to make a prediction for


