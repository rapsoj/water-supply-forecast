import pandas as pd
import numpy as np
import datetime as dt
from datetime import date
import matplotlib.pyplot as plt
import re
import os 
import math

os.chdir("/Users/emilryd/programming/Water Supply Forecast")
def is_float(string):
    if string.size == 0:
        return False
    try:
        float(string)
        return True
    except ValueError:
        return False

# Extracting data from metatadata and train csv files
data = pd.read_csv("Data/site_usgs_ids.csv")
ids = data["site_id"].to_numpy()
names = data["site_name"].to_numpy()
meta_df = pd.read_csv("Data/metadata_TdPVeJC.csv")


names = np.delete(names, [np.where(ids==9080190)])
names = np.delete(names, [np.where(ids==9211150)])

ids = np.delete(ids, [np.where(ids==9080190)]) # Corrupt data for this site, data not retrieved
ids = np.delete(ids, [np.where(ids==9211150)])

# Specify date format
format = '%Y-%m-%d'
clean_data_ids = []
cleaned_data_names = []
for id in ids:
    str_id = str(int(id))
    if len(str_id) < 8:
        str_id = "0" + str_id
    
    id_data = pd.read_csv(f"Data/{str_id}.csv")
    
    keys = id_data.keys()
    only_key = [key for key in keys if "00060-00003" in key] # Extract discharge data
    pattern_str = r'^\d{4}-\d{2}-\d{2}$'

    disch_vals = id_data[only_key].to_numpy()
    if disch_vals.size > 0:
        discharges = [float(discharge) for discharge in disch_vals if not pd.isnull(discharge) and is_float(discharge)]
        
        print(str_id)
        ids = np.array([str(int(i)) for i in meta_df["usgs_id"].to_numpy() if not math.isnan(i)])
        print(ids)
        #shortened_site_name_idx = np.where()[0][0]
        shortened_site_name = meta_df["site_id"].to_numpy()[meta_df["usgs_id"].to_numpy()==id][0]
        print(shortened_site_name)
        dates_list = [dt.datetime.strptime(date, format).date() for idx, date in enumerate(id_data["DateTime"].to_numpy()) if not pd.isnull(disch_vals[idx]) and is_float(disch_vals[idx])]
        #print(id_data[only_key].to_numpy())
        df = pd.DataFrame({"Date": dates_list, "Discharge cubic feet per second":discharges})
        df.to_csv(f"Data/{shortened_site_name}_cleaned.csv")
        clean_data_ids.append(str_id)
        cleaned_data_names.append(names[np.where(ids==id)])
        #print(f"Data Loss: {len(discharges)-len(disch_vals)}. ID = {str_id}")
        
id_df = pd.DataFrame({"site_id": clean_data_ids, "site_name": cleaned_data_names})
id_df.to_csv(f"Data/cleaned_ids.csv")
