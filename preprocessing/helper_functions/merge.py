import os
import csv
import pandas as pd
import numpy as np
from functools import reduce

def merge_site_id_day(list_dataframes) :
    # Merge on site id, day
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['year', 'month', 'day','site_id'],
                                                    how='outer'), list_dataframes)
    return df_merged

def merge_site_id_mon(list_dataframes) :
    # Merge on site id, month
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['year', 'month', 'site_id'],
                                                    how='outer'), list_dataframes)   
    return df_merged

def merge_site_id_year(list_dataframes) :
    # Merge on site id, year 
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['year', 'site_id'],
                                                    how='outer'), list_dataframes)   
    return df_merged

def merge_day(list_dataframes) :
    # Merge on site id, day
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['year', 'month', 'day'],
                                                    how='outer'), list_dataframes)   
    return df_merged

def merge_mon(list_dataframes) :
    # Merge on site id, mon
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['year', 'month'],
                                                    how='outer'), list_dataframes)  
    return df_merged

def merge_year(list_dataframes) :
    # Merge on site id, year
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['year'],
                                                    how='outer'), list_dataframes)  
    return df_merged






# def merge_dfs(data_frames) :
#     df_site_all_dates = []
#     df_site_mon_dates = []
#     df_site_year_dates = []
#     df_all_dates = []
#     df_mon_dates = []
#     df_year_dates = []

#     # Specify the requisite columns
#     site_id_merge = ['site_id']
#     all_dates = ['year', 'month', 'day']
#     mon_dates = ['month', 'year']
#     year_dates = ['year']

#     # Iterate through the list of dataframes
#     for df in data_frames:
#         # Check if the dataframe contains site columns
#         if all(column in df.columns for column in site_id_merge):
            
#             # If contains site columns, check granularity of date
#             if all(column in df.columns for column in all_dates):
#                 df_site_all_dates.append(df)

#             elif all(column in df.columns for column in mon_dates):
#                 df_site_mon_dates.append(df)
            
#             elif all(column in df.columns for column in year_dates):
#                 df_site_year_dates.append(df)

#         # If doesn't contain site columns, check granularity of date
#         if all(column in df.columns for column in all_dates):
#             df_all_dates.append(df)

#         elif all(column in df.columns for column in mon_dates):
#             df_mon_dates.append(df)

#         elif all(column in df.columns for column in year_dates):
#                 df_year_dates.append(df)

#     # Sites merge day
#     try: 
#         df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['year', 'month', 'day', 'site_id'],
#                                                     how='outer'), df_site_all_dates)
#     except: 
#         pass 

#     # Sites merge month
#     merge_list = [df_merged]
#     merge_list.append(df_site_mon_dates) 
    
#     try:
#         df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['year', 'month', 'site_id'],
#                                                     how='outer'), merge_list)
#     except:
#         pass 

#     # Sites merge year
#     merge_list = [df_merged]
#     merge_list.append(df_site_year_dates) 
#     try:
#         df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['year', 'site_id'],
#                                                     how='outer'), merge_list)
#     except:
#         pass 

#     # Days merge
#     merge_list = [df_merged]
#     merge_list.append(df_all_dates) 
#     try:
#         df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['year', 'month', 'day'],
#                                                     how='outer'), df_all_dates)
#     except:
#         pass 

#     # Month merge 
#     merge_list = [df_merged]
#     merge_list.append(df_mon_dates) 
#     try:
#         df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['year', 'month'],
#                                                     how='outer'), merge_list)
#     except:
#         pass 

#     # Year merge
#     merge_list = [df_merged]
#     merge_list.append(df_year_dates) 
#     try:
#         df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['year'],
#                                                     how='outer'), merge_list)
#     except:
#         pass 

#     return(df_merged)
