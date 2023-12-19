import os

import numpy as np
import pandas as pd

from consts import LAST_YEAR
from preprocessing.helper_functions.dictionaries import month_to_num, month_to_num_up


def import_mjo(current_dir):
    # Import mjo dataset
    folder_path = os.path.join(current_dir, '..', 'assets', 'data', 'teleconnections')
    df_mjo = pd.read_table(os.path.join(folder_path, "mjo.txt"), delim_whitespace=True, skiprows=1)
    return df_mjo


def clean_mjo(df_mjo):
    # Basic cleaning for mjo dataset
    df_mjo = df_mjo.iloc[1:]
    df_mjo.columns = df_mjo.columns.str.strip()
    df_mjo = df_mjo.add_prefix('mjo')
    df_mjo = df_mjo[df_mjo['mjo20E'] != '*****']  # Remove future values (missing)
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


def import_nino(current_dir):
    # Import nino dataset
    folder_path = os.path.join(current_dir, '..', 'assets', 'data', 'teleconnections')
    df_nino = pd.read_table(os.path.join(folder_path, "nino_regions_sst.txt"), delim_whitespace=True)
    return df_nino


def clean_nino(df_nino):
    # Basic cleaning for nino dataset
    df_nino = df_nino.rename(columns={'YR': 'year', 'MON': 'month'})
    df_nino = df_nino.rename(columns={c: 'nino' + c for c in df_nino.columns if c not in ['year', 'month']})
    return df_nino


def import_oni(current_dir):
    # Import oni dataset
    folder_path = os.path.join(current_dir, '..', 'assets', 'data', 'teleconnections')
    df_oni = pd.read_table(os.path.join(folder_path, "oni.txt"), delim_whitespace=True)
    return df_oni


def clean_oni(df_oni):
    # Basic cleaning for oni dataset
    df_oni = df_oni.rename(columns={'YR': 'year'})
    df_oni = df_oni.rename(columns={c: 'oni' + c for c in df_oni.columns if c not in ['year', 'month']})

    month_conversion_dictionary = {
        'DJF': 1,
        'JFM': 2,
        'FMA': 3,
        'MAM': 4,
        'AMJ': 5,
        'MJJ': 6,
        'JJA': 7,
        'JAS': 8,
        'ASO': 9,
        'SON': 10,
        'OND': 11,
        'NDJ': 12,
    }
    df_oni['month'] = df_oni.oniSEAS.map(month_conversion_dictionary.get)

    return df_oni.drop(columns='oniSEAS')


def import_pdo(current_dir):
    # Import pdo dataset
    folder_path = os.path.join(current_dir, '..', 'assets', 'data', 'teleconnections')
    df_pdo = pd.read_table(os.path.join(folder_path, "pdo.txt"), delim_whitespace=True, skiprows=1)
    return df_pdo


def clean_pdo(df_pdo):
    # Basic cleaning for pdo dataset
    df_pdo = pd.melt(df_pdo, id_vars=['Year'], var_name='Month', value_name='pdo')
    df_pdo = df_pdo.rename(columns={'Year': 'year', 'Month': 'month'})
    df_pdo['pdo'] = df_pdo['pdo'].replace(99.99, np.nan)  # Remove future values (missing)
    df_pdo['month'] = df_pdo['month'].map(month_to_num)
    return df_pdo


def import_pna(current_dir):
    # Import pna dataset
    folder_path = os.path.join(current_dir, '..', 'assets', 'data', 'teleconnections')
    df_pna = pd.read_table(os.path.join(folder_path, "pna.txt"), delim_whitespace=True)
    return df_pna


def clean_pna(df_pna):
    # Basic cleaning for pna dataset
    df_pna = pd.melt(df_pna, id_vars=['year'], var_name='month', value_name='pna')
    df_pna['month'] = df_pna['month'].map(month_to_num)
    return df_pna


def import_soi1(current_dir):
    # Import soi 1
    folder_path = os.path.join(current_dir, '..', 'assets', 'data', 'teleconnections')
    df_soi1 = pd.read_table(os.path.join(folder_path, "soi1.txt"), delim_whitespace=True, skiprows=3)
    return df_soi1


def import_soi2(current_dir):
    # Import soi 2
    folder_path = os.path.join(current_dir, '..', 'assets', 'data', 'teleconnections')
    df_soi2 = pd.read_table(os.path.join(folder_path, "soi2.txt"), delim_whitespace=True, skiprows=3)
    return df_soi2


def clean_soi1(df_soi1):
    # Clean soi 1
    df_soi1.columns = df_soi1.columns.str.strip()
    df_soi1 = pd.melt(df_soi1, id_vars=['YEAR'], var_name='month', value_name='soi_anom')
    df_soi1 = df_soi1.rename(columns={'YEAR': 'year'})
    df_soi1['month'] = df_soi1['month'].map(month_to_num_up)
    return df_soi1


def clean_soi2(df_soi2):
    # Clean soi 2
    df_soi2.columns = df_soi2.columns.str.strip()
    df_soi2 = pd.melt(df_soi2, id_vars=['YEAR'], var_name='month', value_name='soi_sd')
    df_soi2 = df_soi2.rename(columns={'YEAR': 'year'})
    df_soi2['month'] = df_soi2['month'].map(month_to_num_up)
    return df_soi2


def import_flow(current_dir, additional_sites=False):
    # Import flows training dataset
    if additional_sites:
        folder_path = os.path.join(current_dir, '..', 'assets', 'data', 'additional_sites')
    else:
        folder_path = os.path.join(current_dir, '..', 'assets', 'data')

    df_flow = pd.read_csv(os.path.join(folder_path, "train_monthly_naturalized_flow.csv"))
    df_test_flow = pd.read_csv(os.path.join(folder_path, 'test_monthly_naturalized_flow.csv'))
    df_flow = pd.concat([df_flow, df_test_flow])  # todo fix leakage
    return df_flow


def clean_flow(df_flow):
    # Clean flows training dataset
    return df_flow


def import_grace(current_dir):
    # Import non-pixel grace indicators
    folder_path = os.path.join(current_dir, '..', 'assets', 'data', 'grace_indicators')
    df_grace = pd.read_csv(os.path.join(folder_path, "grace_aggregated.csv"))
    return df_grace


def clean_grace(df_grace):
    # Clean grace dataset
    # Convert 'time' to datetime format
    df_grace['time'] = pd.to_datetime(df_grace['time'])

    # Extract day, month, and year into separate columns
    df_grace['day'] = df_grace['time'].dt.day
    df_grace['month'] = df_grace['time'].dt.month
    df_grace['year'] = df_grace['time'].dt.year
    df_grace.drop('time', axis=1, inplace=True)
    return df_grace


def import_snotel(current_dir):
    # Import Snotel dataset
    folder_path = os.path.join(current_dir, '..', 'assets', 'data')
    df_snotel = pd.read_csv(os.path.join(folder_path, "snotel.csv"))
    return df_snotel


def clean_snotel(df_snotel):
    # Extract day, month, and year into separate columns
    df_snotel['date'] = pd.to_datetime(df_snotel['date'])
    df_snotel['day'] = df_snotel['date'].dt.day
    df_snotel['month'] = df_snotel['date'].dt.month
    df_snotel['year'] = df_snotel['date'].dt.year
    df_snotel.drop('date', axis=1, inplace=True)

    df_snotel = df_snotel.rename(columns={'site': 'site_id'})

    return df_snotel


def import_cpc_prec(current_dir):
    # Import cpc dataset
    folder_path = os.path.join(current_dir, '..', 'assets', 'data', 'cpc_outlooks')
    df_cpc_prec = pd.read_csv(os.path.join(folder_path, "cpc_prec.csv"))
    return df_cpc_prec


def clean_cpc_prec(df_cpc_prec):
    # Convert labels
    df_cpc_prec = df_cpc_prec.drop(columns=['CD', 'R', '90', '50', '10'])
    new_labels = {label: f'{label}_prec' for label in df_cpc_prec.columns}
    df_cpc_prec = df_cpc_prec.rename(columns=new_labels)
    df_cpc_prec = df_cpc_prec.rename(
        columns={'YEAR_prec': 'year', 'MN_prec': 'month', 'site_id_prec': 'site_id', 'LEAD_prec': 'LEAD'})
    df_cpc_prec['day'] = 15

    return df_cpc_prec


def import_cpc_temp(current_dir):
    # Import cpc dataset
    folder_path = os.path.join(current_dir, '..', 'assets', 'data', 'cpc_outlooks')
    df_cpc_prec = pd.read_csv(os.path.join(folder_path, "cpc_temp.csv"))
    return df_cpc_prec


def clean_cpc_temp(df_cpc_temp):
    # Convert labels
    df_cpc_temp = df_cpc_temp.drop(columns=['CD', 'R', '90', '50', '10'])
    new_labels = {label: f'{label}_temp' for label in df_cpc_temp.columns}
    df_cpc_temp = df_cpc_temp.rename(columns=new_labels)
    df_cpc_temp = df_cpc_temp.rename(
        columns={'YEAR_temp': 'year', 'MN_temp': 'month', 'site_id_temp': 'site_id', 'LEAD_temp': 'LEAD'})
    df_cpc_temp['day'] = 15

    return df_cpc_temp


def import_dem(current_dir):
    folder_path = os.path.join(current_dir, '..', 'assets', 'data', 'dem')
    df_dem = pd.read_csv(os.path.join(folder_path, 'dem_summary.csv'))
    return df_dem


def clean_dem(df_dem):
    return df_dem


# todo fix importing and cleaning swann
def import_swann(current_dir, additional_sites):
    if additional_sites:
        folder_path = os.path.join(current_dir, '..', 'assets', 'data', 'additional_sites')
    else:
        folder_path = os.path.join(current_dir, '..', 'assets', 'data', 'swann')
    df_swann = pd.read_csv(os.path.join(folder_path, 'swann_swe.csv'))
    return df_swann


def clean_swann(df_swann, additional_sites=False):
    if additional_sites:
        key = 'week_start_date'
    else:
        key = 'time'
    df_swann[key] = pd.to_datetime(df_swann[key]) + pd.DateOffset(days=7)

    # Extract day, month, and year into separate columns
    df_swann['day'] = df_swann[key].dt.day
    df_swann['month'] = df_swann[key].dt.month
    df_swann['year'] = df_swann[key].dt.year
    df_swann.drop(key, axis=1, inplace=True)
    return df_swann


def import_basins(current_dir, additional_data):
    if additional_data:
        folder_path = os.path.join(current_dir, '..', 'assets', 'data', 'additional_sites')
    else:
        folder_path = os.path.join(current_dir, '..', 'assets', 'data', 'hydrobasins')
    df_basins = pd.read_csv(os.path.join(folder_path, 'hydrobasins_summary.csv'))
    return df_basins


def clean_basins(df_basins):
    return df_basins


def import_acis(current_dir):
    folder_path = os.path.join(current_dir, '..', 'assets', 'data', 'acis')
    df_acis = pd.read_csv(os.path.join(folder_path, 'acis.csv'))
    return df_acis


def clean_acis(df_acis):
    df_acis.week_start_date = pd.to_datetime(df_acis.week_start_date) + pd.DateOffset(days=7)
    df_acis['day'] = df_acis['week_start_date'].dt.day
    df_acis['month'] = df_acis['week_start_date'].dt.month
    df_acis['year'] = df_acis['week_start_date'].dt.year
    df_acis.drop('week_start_date', axis=1, inplace=True)
    df_acis = df_acis[df_acis.year <= LAST_YEAR]
    return df_acis


def import_pdsi(current_dir):
    folder_path = os.path.join(current_dir, '..', 'assets', 'data', 'pdsi')
    df_pdsi = pd.read_csv(os.path.join(folder_path, 'pdsi_summary.csv'))
    return df_pdsi


def clean_pdsi(df_pdsi):
    df_pdsi['date'] = pd.to_datetime(df_pdsi['date'])
    df_pdsi['day'] = df_pdsi['date'].dt.day
    df_pdsi['month'] = df_pdsi['date'].dt.month
    df_pdsi['year'] = df_pdsi['date'].dt.year
    df_pdsi.drop('date', axis=1, inplace=True)
    return df_pdsi


def import_era5(current_dir):
    folder_path = os.path.join(current_dir, '..', 'assets', 'data', 'ERA5L')
    df_era5 = pd.read_csv(os.path.join(folder_path, 'era5l.csv'))
    return df_era5


def clean_era5(df_era5):
    df_era5['period_end'] = pd.to_datetime(df_era5['period_end']) + pd.DateOffset(days=1)
    df_era5['day'] = df_era5['period_end'].dt.day
    df_era5['month'] = df_era5['period_end'].dt.month
    df_era5['year'] = df_era5['period_end'].dt.year
    df_era5.drop(columns=['period_end', 'period_start'], inplace=True)
    return df_era5


def import_usgs(current_dir):
    folder_path = os.path.join(current_dir, '..', 'assets', 'data', 'usgs_streamflow')
    df_usgs = pd.read_csv(os.path.join(folder_path, 'usgs_streamflow.csv'))
    return df_usgs


def clean_usgs(df_usgs):
    df_usgs.week_start_date = pd.to_datetime(df_usgs.week_start_date) + pd.DateOffset(days=7)
    df_usgs['day'] = df_usgs['week_start_date'].dt.day
    df_usgs['month'] = df_usgs['week_start_date'].dt.month
    df_usgs['year'] = df_usgs['week_start_date'].dt.year
    df_usgs.drop('week_start_date', axis=1, inplace=True)
    df_acis = df_usgs[df_usgs.year <= LAST_YEAR]
    return df_usgs
