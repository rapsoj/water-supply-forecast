import pandas as pd
import matplotlib.pyplot as plt
import os
from preprocessing.generic_preprocessing import get_processed_dataset
from preprocessing.pre_ml_processing import ml_preprocess_data
os.chdir("../exploration")
global_mjo_cols = ['mjo20E', 'mjo70E', 'mjo80E', 'mjo100E', 'mjo120E', 'mjo140E', 'mjo160E', 'mjo120W', 'mjo40W',
                   'mjo10W']
global_oni_cols = ['oniTOTAL', 'oniANOM']
global_nino_cols = ['ninoNINO1+2', 'ninoANOM', 'ninoNINO3', 'ninoANOM.1', 'ninoNINO4', 'ninoANOM.2',
                    'ninoNINO3.4', 'ninoANOM.3']
global_misc_cols = ['pdo', 'pna', 'soi_anom', 'soi_sd']

shared_cols = ['date']



def sanity_function(ini_data: pd.DataFrame, processed_data: pd.DataFrame, col: str):
    # plt.scatter(ini_data.date.iloc[::100][~ini_data[col].iloc[::100].isna()], ini_data[col].iloc[::100].dropna(),
    #            c='b')
    # plt.scatter(processed_data.date.iloc[::100][~processed_data[col].iloc[::100].isna()],
    #            processed_data[col].iloc[::100].dropna(), c='r')
    # monthly measurements are approx at the middle

    # To check global variables, remove the [ini_data.site_id == site_id] in the ini_data scatter plot,
    # for local variables keep it. Site id picked at the discretion of the programmer (YOU).
    site_id = "animas_r_at_durango"
    plt.scatter(ini_data.date[~ini_data[col].isna()][ini_data.site_id == site_id],
                ini_data[col].dropna()[ini_data.site_id == site_id], alpha=0.7, c='b')
    plt.plot(processed_data.date[processed_data.site_id == site_id].iloc[::1],
                processed_data[col][processed_data.site_id == site_id].iloc[::1], alpha=0.7, c='g')

    plt.title(col)
    plt.show()
    plt.pause(1e-3)

def fix_df(data: pd.DataFrame):
    # Create dates to work with
    ini_data = data
    ini_data.day[ini_data.day == -1] = 15

    date_cols = ['year', 'month', 'day']

    ini_data["date"] = pd.to_datetime(ini_data[date_cols].map(int))
    site_id_str = 'site_id_'
    site_id_cols = [col for col in data.columns if 'site_id' in col]

    ini_data['site_id'] = ini_data[site_id_cols] \
        .idxmax(axis='columns') \
        .apply(lambda x: x[x.find(site_id_str) + len(site_id_str):])
    ini_data = ini_data.sort_values('date')

    # todo make sure you re-incorporate all of these+interpolate them properly
    mjo_data = ini_data[global_mjo_cols + shared_cols].dropna()
    oni_data = ini_data[global_oni_cols + shared_cols].dropna()

    nino_data = ini_data[global_nino_cols + shared_cols].dropna()
    misc_data = ini_data[global_misc_cols + shared_cols].dropna()
    # Keeping only SNOTEL data
    ini_data = ini_data.drop(columns=global_mjo_cols + global_nino_cols + global_oni_cols + global_misc_cols)
    ini_data = ini_data.merge(mjo_data, on='date', how='outer') \
        .merge(nino_data.drop_duplicates(), on='date', how='outer') \
        .merge(oni_data.drop_duplicates(), on='date', how='outer') \
        .merge(misc_data.drop_duplicates(), on='date', how='outer') \
        .sort_values(by='date') \
        .reset_index(drop=True)

    return ini_data

def analyze_data():
    basic_preprocessed_df = get_processed_dataset(load_from_cache=True)

    # todo add explicit forecasting functionality, split train/test for forecasting earlier.
    #  currently everything is processed together. unsure if necessary
    processed_data = ml_preprocess_data(basic_preprocessed_df, load_from_cache=True)

    ini_data = fix_df(basic_preprocessed_df)
    for col in processed_data.columns:
        sanity_function(ini_data, processed_data, col)

if __name__ == '__main__':
    analyze_data()