import os
import csv
import pandas as pd
from functools import reduce
import calendar
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

def import_mjo(current_dir) :
    # Import mjo dataset
    folder_path = os.path.join(current_dir, '..', 'assets', 'data', 'teleconnections')
    df_mjo = pd.read_table(os.path.join(folder_path,"mjo.txt"),delim_whitespace=True, skiprows=1)
    return df_mjo

def clean_mjo(df_mjo): 
    # Basic cleaning for mjo dataset
    df_mjo = pd.read_table(os.path.join(folder_path,"mjo.txt"),delim_whitespace=True, skiprows=1)
    df_mjo = df_mjo.iloc[1:]
    df_mjo.columns = df_mjo.columns.str.strip()
    df_mjo = df_mjo.add_prefix('mjo')
    df_mjo = df_mjo[df_mjo['mjo20E'] != '*****'] # Remove future values (missing)
    # Iterate over columns
    for column_name in df_mjo.columns:
        # Check if the column name contains the substring 'mjo'
        if 'mjo' in column_name:
            # Convert values to float using pd.to_numeric
            df_mjo[column_name] = pd.to_numeric(df_mjo[column_name], errors='coerce')
    df_mjo['year'] = df_mjo['mjoPENTAD'].astype(str).str[:4].astype(int)
    df_mjo['month'] = df_mjo['mjoPENTAD'].astype(str).str[4:6].astype(int)
    df_mjo['day'] = df_mjo['mjoPENTAD'].astype(str).str[6:8].astype(int)

    df_mjo = df_mjo.drop(columns='mjoPENTAD')
    return df_mjo

def import_nino(current_dir) :
    # Import nino dataset
    folder_path = os.path.join(current_dir, '..','assets', 'data', 'teleconnections')
    df_nino = pd.read_table(os.path.join(folder_path,"nino_regions_sst.txt"),delim_whitespace=True)
    return df_nino

def clean_nino(df_nino) :
    # Basic cleaning for nino dataset
    df_nino = df_nino.rename(columns={'YR':'year', 'MON':'month'})
    df_nino = df_nino.rename(columns={c: 'nino'+c for c in df_nino.columns if c not in ['year', 'month']})
    df_nino['day'] = -1
    return df_nino

def import_oni(current_dir) :
    # Import oni dataset
    folder_path = os.path.join(current_dir, '..','assets', 'data', 'teleconnections')
    df_oni = pd.read_table(os.path.join(folder_path,"oni.txt"),delim_whitespace=True)
    return df_oni

def clean_oni(df_oni) :
    # Basic cleaning for oni dataset
    df_oni = df_oni.rename(columns={'YR':'year'})
    df_oni['month'] = -1 #Assume month of collection is january
    df_oni = df_oni.rename(columns={c: 'oni'+c for c in df_oni.columns if c not in ['year', 'month']})
    df_oni['day'] = -1
    return df_oni

def import_pdo(current_dir) :
    # Import pdo dataset
    folder_path = os.path.join(current_dir, '..','assets', 'data', 'teleconnections')
    df_pdo = pd.read_table(os.path.join(folder_path,"pdo.txt"),delim_whitespace=True,skiprows=1)

def clean_pdo(df_pdo) :
    # Basic cleaning for pdo dataset
    df_pdo = pd.melt(df_pdo, id_vars=['Year'], var_name='Month', value_name='pdo')
    df_pdo = df_pdo.rename(columns={'Year':'year', 'Month':'month'})
    df_pdo = df_pdo[df_pdo['pdo'] != 99.99] # Remove future values (missing)
    df_pdo['month'] = df_pdo['month'].map(month_to_num)
    df_pdo['day'] = -1

def import_pna(current_dir) :
    # Import pna dataset
    folder_path = os.path.join(current_dir, '..','assets', 'data', 'teleconnections')
    df_pna = pd.read_table(os.path.join(folder_path,"pna.txt"),delim_whitespace=True)

def clean_pna(df_pna) :
    # Basic cleaning for pna dataset
    df_pna = pd.melt(df_pna, id_vars=['year'], var_name='month', value_name='pna')
    df_pna['month'] = df_pna['month'].map(month_to_num)
    df_pna['day'] = -1

def import_soi1(current_dir) :
    # Import soi 1
    folder_path = os.path.join(current_dir, '..','assets', 'data', 'teleconnections')
    df_soi1 = pd.read_table(os.path.join(folder_path,"soi1.txt"),delim_whitespace=True,skiprows=3)

def import_soi2(current_dir) :
    # Import soi 2
    folder_path = os.path.join(current_dir, '..','assets', 'data', 'teleconnections')
    df_soi2 = pd.read_table(os.path.join(folder_path,"soi2.txt"),delim_whitespace=True,skiprows=3)

def clean_soi1(df_soi1) :
    # Clean soi 1
    df_soi1.columns = df_soi1.columns.str.strip()
    df_soi1 = pd.melt(df_soi1, id_vars=['YEAR'], var_name='month', value_name='soi_anom')
    df_soi1 = df_soi1.rename(columns={'YEAR':'year'})
    df_soi1['month'] = df_soi1['month'].map(month_to_num_up)
    df_soi1['day'] = -1

def clean_soi2(df_soi2) :
    # Clean soi 2
    df_soi2.columns = df_soi2.columns.str.strip()
    df_soi2 = pd.melt(df_soi2, id_vars=['YEAR'], var_name='month', value_name='soi_sd')
    df_soi2 = df_soi2.rename(columns={'YEAR':'year'})
    df_soi2['month'] = df_soi2['month'].map(month_to_num_up)
    df_soi2['day'] = -1

def import_flow(current_dir) :
    # Import flows training dataset
    folder_path = os.path.join(current_dir, '..','assets', 'data')
    df_flow = pd.read_csv(os.path.join(folder_path,"train_monthly_naturalized_flow.csv"))

def clean_flow(df_flow) :
    # Clean flows training dataset
    df_flow['day'] = -1

def import_grace(current_dir) :
    # Import non-pixel grace indicators
    folder_path = os.path.join(current_dir, '..','assets', 'data', 'grace_indicators')
    df_grace = pd.read_csv(os.path.join(folder_path,"grace_aggregated.csv"))

def clean_grace(df_grace) :
    # Clean grace dataset
    # Convert 'time' to datetime format
    df_grace['time'] = pd.to_datetime(df_grace['time'])

    # Extract day, month, and year into separate columns
    df_grace['day'] = df_grace['time'].dt.day
    df_grace['month'] = df_grace['time'].dt.month
    df_grace['year'] = df_grace['time'].dt.year

    df_grace.drop('time', axis=1, inplace=True)

    data_frames = [df_merged, df_grace]
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['year', 'month', 'day', 'site_id'],
                                                how='outer'), data_frames)
    
def import_snotel(current_dir) :
    # Import Snotel dataset
    df_snotel = pd.read_csv(os.path.join(current_dir,"snotel.csv"))

def clean_snotel(df_snotel) :
    # Extract day, month, and year into separate columns
    df_snotel['date'] = pd.to_datetime(df_snotel['date'])
    df_snotel['day'] = df_snotel['date'].dt.day
    df_snotel['month'] = df_snotel['date'].dt.month
    df_snotel['year'] = df_snotel['date'].dt.year
    df_snotel.drop('date', axis=1, inplace=True)

    df_snotel= df_snotel.rename(columns={'site':'site_id'})
