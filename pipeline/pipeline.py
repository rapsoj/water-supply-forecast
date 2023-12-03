import numpy as np
import pandas as pd

from benchmark.benchmark_results import benchmark_results, cache_preds
from models.fit_to_data import gen_basin_preds
from preprocessing.generic_preprocessing import get_processed_dataset
from preprocessing.pre_ml_processing import ml_preprocess_data


def run_pipeline(test_years: tuple = tuple(np.arange(2005, 2024, 2)),
                 validation_years: tuple = tuple(np.arange(1984, 2023, 8)), load_from_cache: bool = False):
    print('Loading data')
    # todo add output_csv paths to preprocessing, especially the ml preprocessing
    basic_preprocessed_df = get_processed_dataset(load_from_cache=load_from_cache)

    # todo add explicit forecasting functionality, split train/test for forecasting earlier.
    #  currently everything is processed together. unsure if necessary
    processed_data, processed_ground_truth = ml_preprocess_data(basic_preprocessed_df, load_from_cache=load_from_cache)

    # Get training, validation and test sets
    train_features, val_features, test_features, train_gt, val_gt, test_gt = \
        train_val_test_split(processed_data, processed_ground_truth, test_years, validation_years)

    # todo implement global models
    site_ids = processed_data.site_id.unique()
    non_feat_cols = ['date', 'site_id', 'forecast_year']
    for site_id in site_ids:
        print(f'Fitting to site {site_id}')
        train_site = train_features[train_features.site_id == site_id].drop(columns=non_feat_cols)
        val_site = val_features[val_features.site_id == site_id].drop(columns=non_feat_cols)
        test_site = test_features[test_features.site_id == site_id]
        test_dates = test_site.date
        test_site = test_site.drop(columns=non_feat_cols)

        #num_predictions = train_site.groupby('forecast_year').size()


        train_pred, val_pred, test_pred = gen_basin_preds(train_site, train_gt, val_site, val_gt, test_site)

        results_id = f'{site_id}'
        print(f'Benchmarking results for site {site_id}')
        train_pred, val_pred, test_pred = benchmark_results(train_pred, train_gt, val_pred, val_gt, test_pred,
                                                            benchmark_id=results_id)
        cache_preds(pred=test_pred, cache_id=results_id, site_id=site_id, pred_dates=test_dates)




def train_val_test_split(feature_df: pd.DataFrame, gt_df: pd.DataFrame, test_years: tuple, validation_years: tuple):
    feature_df = feature_df.copy()
    gt_df = gt_df.copy()

    test_feature_mask = feature_df.forecast_year.isin(test_years)
    test_gt_mask = gt_df.forecast_year.isin(test_years)

    test_feature_df = feature_df[test_feature_mask].drop(columns='volume').reset_index(drop=True)
    test_gt_df = gt_df[test_gt_mask].reset_index(drop=True)

    validation_feature_mask = feature_df.forecast_year.isin(validation_years)
    validation_gt_mask = gt_df.forecast_year.isin(validation_years)

    val_feature_df = feature_df[validation_feature_mask].reset_index(drop=True)
    val_gt_df = gt_df[validation_gt_mask].reset_index(drop=True)

    train_feature_df = feature_df[~validation_feature_mask & ~test_feature_mask]
    train_gt_df = gt_df[~validation_gt_mask & ~test_gt_mask]

    assert train_feature_df.date.isin(val_feature_df.date).sum() == 0 and \
           train_feature_df.date.isin(test_feature_df.date).sum() == 0 and \
           val_feature_df.date.isin(test_feature_df.date).sum() == 0, \
        "Dates are overlapping between train, val, and test sets"

    return train_feature_df, val_feature_df, test_feature_df, train_gt_df, val_gt_df, test_gt_df


if __name__ == '__main__':
    run_pipeline()
