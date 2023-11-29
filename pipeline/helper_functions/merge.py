import os
import csv
import pandas as pd
import numpy as np
from functools import reduce

def merge_dfs(data_frames) :
    df_site_all_dates = []
    df_site_mon_dates = []
    df_site_year_dates = []
    df_all_dates = []
    df_mon_dates = []
    df_year_dates = []

    # Specify the requisite columns
    site_id_merge = ['site_id']
    all_dates = ['year', 'month', 'day']
    mon_dates = ['month', 'year']
    year_dates = ['year']

    # Iterate through the list of dataframes
    for df in data_frames:
        # Check if the dataframe contains all required columns
        if all(column in df.columns for column in site_id_merge):

            if all(column in df.columns for column in all_dates):
                df_site_all_dates.append(df)

            if all(column in df.columns for column in mon_dates):
                df_site_mon_dates.append(df)
            
            else :
                df_site_year_dates.append(df)

        if all(column in df.columns for column in all_dates):
            df_all_dates.append(df)

        if all(column in df.columns for column in mon_dates):
            df_mon_dates.append(df)

        else : 
            df_year_dates.append(df)

    # Sites merge day
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['year', 'month', 'day', 'site_id'],
                                                    how='outer'), df_site_all_dates)

    # Sites merge month
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['year', 'month', 'site_id'],
                                                    how='outer'), df_merged, df_site_mon_dates)

    # Sites merge year
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['year', 'site_id'],
                                                    how='outer'), df_merged, df_site_year_dates)

    # Days merge
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['year', 'month', 'day'],
                                                    how='outer'), df_all_dates)
    # Month merge 
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['year', 'month'],
                                                    how='outer'), df_merged, df_mon_dates)
    # Year merge
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['year'],
                                                    how='outer'), df_merged, df_year_dates)
    