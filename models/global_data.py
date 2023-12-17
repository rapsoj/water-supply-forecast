import pickle

import pandas as pd
import os
from preprocessing.pre_ml_processing import ml_preprocess_data
from preprocessing.generic_preprocessing import get_processed_dataset
from consts import N_PRED_MONTHS, N_PREDS_PER_MONTH, FIRST_FULL_GT_YEAR


def load_ground_truth(num_predictions: int):
    ground_truth_df = pd.read_csv(os.path.join("..", "assets", "data", "train.csv"))
    # todo improve how we retrieve data for different sites, retrieving as much data as we can for each
    year_mask = (ground_truth_df.year >= FIRST_FULL_GT_YEAR)
    ground_truth_df = ground_truth_df[year_mask].reset_index(drop=True)
    ground_truth_df = ground_truth_df.loc[ground_truth_df.index.repeat(num_predictions)]
    ground_truth_df['forecast_year'] = ground_truth_df.year
    ground_truth_df = ground_truth_df.drop(columns=['year'])

    return ground_truth_df


def train_val_test_split(feature_df: pd.DataFrame, gt_df: pd.DataFrame, test_years: tuple, validation_years: tuple,
                         start_year: int = FIRST_FULL_GT_YEAR):
    feature_df = feature_df.copy()
    gt_df = gt_df.copy()

    test_feature_mask = feature_df.forecast_year.isin(test_years)
    test_gt_mask = gt_df.forecast_year.isin(test_years)

    test_feature_df = feature_df[test_feature_mask].reset_index(drop=True)

    val_mask = feature_df.forecast_year.isin(validation_years)
    val_gt_mask = gt_df.forecast_year.isin(validation_years)

    val_feature_df = feature_df[val_mask].reset_index(drop=True)
    val_gt_df = gt_df[val_gt_mask].reset_index(drop=True)

    train_mask = ~val_mask & ~test_feature_mask & (feature_df.forecast_year >= start_year)
    train_gt_mask = ~val_gt_mask & ~test_gt_mask & (gt_df.forecast_year >= start_year)

    # todo filter this in a more dynamic way, getting the minimum year
    train_feature_df = feature_df[train_mask]
    train_gt_df = gt_df[train_gt_mask]

    assert train_feature_df.date.isin(val_feature_df.date).sum() == 0 and \
           train_feature_df.date.isin(test_feature_df.date).sum() == 0 and \
           val_feature_df.date.isin(test_feature_df.date).sum() == 0, \
        "Dates are overlapping between train, val, and test sets"

    # todo figure out why some things are empty here, e.g. test_gt_df
    return train_feature_df, val_feature_df, test_feature_df, train_gt_df, val_gt_df


def get_global_data():
    load_from_cache = True
    basic_preprocessed_df = get_processed_dataset(load_from_cache=load_from_cache)

    processed_data = ml_preprocess_data(basic_preprocessed_df, load_from_cache=load_from_cache)

    assert processed_data.isna().any().sum() == 1, 'More than 1 column has Nans. Should only be the volume column.'
    processed_data = processed_data.fillna(0)

    test_years = range(2005, 2024, 2)
    validation_years = range(FIRST_FULL_GT_YEAR, 2023, 8)

    gt = load_ground_truth(N_PRED_MONTHS * N_PREDS_PER_MONTH)

    X, val_X, test_X, y, val_y = train_val_test_split(processed_data, gt, test_years, validation_years)

    return X, val_X, test_X, y, val_y


res = get_global_data()
with open('global_data.pkl', 'wb') as f:
    pickle.dump(res, f)
