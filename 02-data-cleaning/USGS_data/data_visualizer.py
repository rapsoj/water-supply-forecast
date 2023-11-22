import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from astropy.timeseries import LombScargle
import astropy.units as u
import os
format = '%Y-%m-%d'

os.chdir("/Users/emilryd/programming/water-supply-forecast/02-data-cleaning/USGS_data")
# get cleaned ids
df = pd.read_csv("Data/cleaned_ids.csv")
site_ids = pd.read_csv("Data/site_usgs_ids.csv")["site_id"].to_numpy()
sites = pd.read_csv("Data/site_usgs_ids.csv").to_numpy()[:,1]
ids = df.to_numpy()[:,1]
for idx, id in enumerate(ids):
    # get cleaned data
    str_id = str(id)
    print(str_id)
    if len(str_id) < 8:
        str_id = "0" + str_id
    data = pd.read_csv(f"Data/{str_id}_cleaned.csv")
    dates_list = [dt.datetime.strptime(date, format).date() for idx, date in enumerate(data["Date"].to_numpy())]
    keys = data.keys()
    plt.plot(dates_list, data[keys[2]])
    plt.xlabel(keys[1])
    plt.ylabel(keys[2])
    plt.title(f"{str_id}, {sites[np.where(site_ids==id)]}")
    plt.show()
    frequency, power = LombScargle(data[keys[0]].to_numpy()/365, data[keys[2]], normalization='psd').autopower(maximum_frequency=5)
    plt.plot(frequency, power)
    plt.xlabel("Frequency (1/years)")
    plt.ylabel("Approximate Fourier PSD")
    plt.title(f"{str_id}, {sites[np.where(site_ids==id)]}")
    plt.show()

