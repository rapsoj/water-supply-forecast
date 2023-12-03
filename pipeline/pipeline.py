import numpy as np
import pandas as pd

from benchmark.benchmark_results import benchmark_results, cache_preds
from models.fit_to_data import gen_basin_preds
from preprocessing.generic_preprocessing import get_processed_dataset
from preprocessing.pre_ml_processing import ml_preprocess_data


def run_pipeline(gt_col: str = 'volume', test_years: tuple = tuple(np.arange(2003, 2024, 2)),
                 validation_years: tuple = tuple(np.arange(1984, 2023, 8)), load_from_cache: bool = True):
    # todo add output_csv paths to preprocessing, especially the ml preprocessing
    basic_preprocessed_df = get_processed_dataset(load_from_cache=load_from_cache)

    # todo add explicit forecasting functionality, split train/test for forecasting earlier.
    #  currently everything is processed together. unsure if necessary
    processed_data = ml_preprocess_data(basic_preprocessed_df, load_from_cache=load_from_cache)

    # Get training, validation and test sets
    train, val, test = train_val_test_split(processed_data, test_years, validation_years)

    assert gt_col not in test.columns, 'Error - test should not have a ground truth!'

    # todo implement global models
    site_ids = processed_data.site_id.unique()
    non_feat_cols = ['date', 'site_id', 'forecast_year']
    for site_id in site_ids:
        train_site = train[train.site_id == site_id].drop(columns=non_feat_cols)
        val_site = val[val.site_id == site_id].drop(columns=non_feat_cols)
        test_site = test[test.site_id == site_id].drop(columns=non_feat_cols)

        train_gt = train_site[gt_col]
        train_site = train_site.drop(columns=gt_col)
        val_gt = val_site[gt_col]
        val_site = val_site.drop(columns=gt_col)

        train_pred, val_pred, test_pred = gen_basin_preds(train_site, train_gt, val_site, val_gt, test_site)

        results_id = f'{site_id}'
        train_pred, val_pred, test_pred = benchmark_results(train_pred, train_gt, val_pred, val_gt, test_pred,
                                                            benchmark_id=results_id)
        cache_preds(pred=test_pred, cache_id=results_id, site_id=site_id, pred_dates=test_site.date)


def train_val_test_split(df: pd.DataFrame, test_years: list, validation_years: list):
    df = df.copy()
    test_mask = df.forecast_year.isin(test_years)
    test_df = df[test_mask].drop(columns='volume').reset_index(drop=True)
    df = df.drop(test_df.index)

    validation_mask = df.forecast_year.isin(validation_years)
    val_df = df[validation_mask].reset_index(drop=True)
    train_df = df.drop(val_df.index).reset_index(drop=True)

    assert train_df.date.isin(val_df.date).sum() == 0 and \
           train_df.date.isin(test_df.date).sum() == 0 and \
           val_df.date.isin(test_df.date).sum() == 0, "Dates are overlapping between train, val, and test sets"

    return train_df, val_df, test_df


if __name__ == '__main__':
    run_pipeline()
