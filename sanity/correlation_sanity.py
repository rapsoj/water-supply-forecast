import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.generic_preprocessing import get_processed_dataset
from preprocessing.pre_ml_processing import ml_preprocess_data
from pipeline.pipeline import get_processed_data_and_ground_truth
from consts import N_PRED_MONTHS, N_PREDS_PER_MONTH, FIRST_FULL_GT_YEAR, JULY, TEST_YEARS
from models.fitters import base_feature_adapter
from scipy.stats import pearsonr
import numpy as np
import seaborn as sns

def analyze_data_sitewise():

    df, ground_truth = get_processed_data_and_ground_truth()
    processed_data = df[~df.forecast_year.isin(TEST_YEARS) & (df.date.dt.month <= JULY)].reset_index(drop=True)
    gt_df = ground_truth.reset_index(drop=True)
    site_id = "animas_r_at_durango"
    processed_data = processed_data.sort_values(by='date')
    processed_data = processed_data[processed_data.site_id == site_id]

    processed_data['year'] = processed_data.date.dt.year
    processed_data = processed_data.drop(columns=['site_id'])

    processed_data = base_feature_adapter(processed_data)

    #processed_data = processed_data.groupby('year').agg(lambda x: x.mean())

    df = pd.DataFrame(columns=['ground_truth'])
    df['ground_truth'] = pd.Series(gt_df[gt_df.site_id == site_id].volume.reset_index(drop=True).unique())
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
    plt.plot(l_corr['gt'])
    plt.show()
    #sns.heatmap(corr_matrix, annot=True, mask=l_corr, annot_kws={"fontsize":6}, xticklabels=True, yticklabels=True, cmap='viridis')
    #plt.show()

def analyze_data_globally(show_heatmap = False):


    df, ground_truth = get_processed_data_and_ground_truth()
    df = df[~df.forecast_year.isin(TEST_YEARS) & (df.date.dt.month <= JULY)].reset_index(drop=True)
    ground_truth = ground_truth.reset_index(drop=True)
    df['ground_truth'] = ground_truth.volume

    drop_cols = ['site_id', 'date', 'forecast_year']

    df.drop(drop_cols, axis=1, inplace=True)
    corr_matrix = df.corr()
    if show_heatmap:
        l_corr = np.triu(corr_matrix)
        sns.heatmap(corr_matrix, annot=True, mask=l_corr, annot_kws={"fontsize":6}, xticklabels=True, yticklabels=True, cmap='viridis')
        plt.show()

    indices = corr_matrix.index[corr_matrix['ground_truth'].abs() >= 0.3]

    for column in indices:
        plt.scatter(df[column], ground_truth.volume)
        plt.title(column)
        plt.show()


if __name__ == '__main__':
    analyze_data_globally()
    analyze_data_sitewise()