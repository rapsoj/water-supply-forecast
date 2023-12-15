import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.generic_preprocessing import get_processed_dataset
from preprocessing.pre_ml_processing import ml_preprocess_data
from pipeline.pipeline import load_ground_truth
from consts import N_PRED_MONTHS, N_PREDS_PER_MONTH, FIRST_FULL_GT_YEAR
from models.models import base_feature_adapter

def analyze_data():
    basic_preprocessed_df = get_processed_dataset(load_from_cache=True)

    # todo add explicit forecasting functionality, split train/test for forecasting earlier.
    #  currently everything is processed together. unsure if necessary
    processed_data = ml_preprocess_data(basic_preprocessed_df, load_from_cache=True)
    processed_data = base_feature_adapter(processed_data)
    test_years = range(2005, 2024, 2)
    processed_data = processed_data[(processed_data.forecast_year >= FIRST_FULL_GT_YEAR) & (processed_data.date.dt.month <= 7) & ~(processed_data.date.dt.year.isin(test_years))]
    site_id = "animas_r_at_durango"
    processed_data = processed_data.sort_values(by='date')
    processed_data = processed_data[processed_data.site_id == site_id]

    processed_data['year'] = processed_data.date.dt.year
    processed_data = processed_data.drop(columns=['date', 'site_id'])

    processed_data = processed_data.groupby('year').agg(lambda x: x.mean())
    for col in processed_data.columns:
        gt_df = load_ground_truth(N_PRED_MONTHS*N_PREDS_PER_MONTH)
        gt_df = gt_df.sort_values(by='forecast_year')

        plt.scatter(processed_data[col], gt_df[gt_df.site_id==site_id].volume.unique())
        plt.xlabel("Feature")
        plt.ylabel("Ground truth (volume)")
        plt.title(f"{col}")
        plt.show()



if __name__ == '__main__':
    analyze_data()