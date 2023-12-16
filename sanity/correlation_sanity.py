import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.generic_preprocessing import get_processed_dataset
from preprocessing.pre_ml_processing import ml_preprocess_data
from pipeline.pipeline import load_ground_truth
from consts import N_PRED_MONTHS, N_PREDS_PER_MONTH, FIRST_FULL_GT_YEAR
from models.models import base_feature_adapter
from scipy.stats import pearsonr
import numpy as np
import seaborn as sns

def analyze_data():
    basic_preprocessed_df = get_processed_dataset(load_from_cache=True)

    # todo add explicit forecasting functionality, split train/test for forecasting earlier.
    #  currently everything is processed together. unsure if necessary
    processed_data = ml_preprocess_data(basic_preprocessed_df, load_from_cache=True)
    test_years = range(2005, 2024, 2)
    processed_data = processed_data[(processed_data.forecast_year >= FIRST_FULL_GT_YEAR) & (processed_data.date.dt.month <= 7) & ~(processed_data.date.dt.year.isin(test_years))]
    site_id = "animas_r_at_durango"
    processed_data = processed_data.sort_values(by='date')
    processed_data = processed_data[processed_data.site_id == site_id]

    processed_data['year'] = processed_data.date.dt.year
    processed_data = processed_data.drop(columns=['site_id'])

    processed_data = base_feature_adapter(processed_data)

    gt_df = load_ground_truth(N_PRED_MONTHS * N_PREDS_PER_MONTH)
    gt_df = gt_df.sort_values(by='forecast_year')



    #processed_data = processed_data.groupby('year').agg(lambda x: x.mean())
    df = pd.DataFrame(columns=['gt'])
    df['gt'] = pd.Series(gt_df[gt_df.site_id == site_id].volume.reset_index(drop=True).unique())
    y = gt_df[gt_df.site_id == site_id].volume.reset_index(drop=True)
    for col in processed_data.columns:
        if len(processed_data[col].unique()) > 1:
            df[col] = processed_data[col].reset_index(drop=True)

        x = processed_data[col].reset_index(drop=True)

        plt.scatter(x, y)
        plt.xlabel("Feature")
        plt.ylabel("Ground truth (volume)")
        plt.title(f"{col}")
        plt.show()

    corr_matrix = df.corr()
    l_corr = np.triu(corr_matrix)
    sns.heatmap(corr_matrix, annot=True, mask=l_corr, annot_kws={"fontsize":6}, xticklabels=True, yticklabels=True, cmap='viridis')
    plt.show()


if __name__ == '__main__':
    analyze_data()