import pandas as pd
import matplotlib.pyplot as plt
import os
from preprocessing.generic_preprocessing import get_processed_dataset
from preprocessing.pre_ml_processing import ml_preprocess_data

path = os.getcwd()


def sanity_function(ini_data: pd.DataFrame, processed_data: pd.DataFrame, col: str):
    # plt.scatter(ini_data.date.iloc[::100][~ini_data[col].iloc[::100].isna()], ini_data[col].iloc[::100].dropna(),
    #            c='b')
    # plt.scatter(processed_data.date.iloc[::100][~processed_data[col].iloc[::100].isna()],
    #            processed_data[col].iloc[::100].dropna(), c='r')
    # monthly measurements are approx at the middle


    site_id = "animas_r_at_durango"
    plt.scatter(ini_data.date[ini_data.site_id == site_id].iloc[::10],
                ini_data[col][ini_data.site_id == site_id].iloc[::10], c='b')
    plt.scatter(processed_data.date[processed_data.site_id == site_id].iloc[::10],
                processed_data[col][processed_data.site_id == site_id].iloc[::10], c='r')

    plt.title(col)
    plt.show()

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