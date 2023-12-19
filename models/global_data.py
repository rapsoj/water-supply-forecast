import os
import pickle

from consts import N_PRED_MONTHS, N_PREDS_PER_MONTH, FIRST_FULL_GT_YEAR, N_PREDS
from pipeline.pipeline import load_ground_truth, train_val_test_split
from preprocessing.generic_preprocessing import get_processed_dataset
from preprocessing.pre_ml_processing import ml_preprocess_data
from preprocessing.data_pruning import data_pruning


def get_global_data():
    load_from_cache = False
    basic_preprocessed_df = get_processed_dataset(load_from_cache=load_from_cache)

    processed_data = ml_preprocess_data(basic_preprocessed_df, load_from_cache=load_from_cache)

    assert processed_data.isna().any().sum() == 4, \
        'Not only volume, wet_cl_smj (@pecos_r_nr_pecos), 00060_Mean (@4 sites), and avgt_d (@3 sites) have Nans'
    processed_data = processed_data.fillna(0)

    test_years = range(2005, 2024, 2)
    validation_years = range(FIRST_FULL_GT_YEAR, 2023, 8)

    gt = load_ground_truth(N_PREDS)

    pruned_data = data_pruning(processed_data, ground_truth=gt)

    X, val_X, test_X, y, val_y = train_val_test_split(pruned_data, gt, test_years, validation_years)

    y = y.sort_values(['site_id', 'forecast_year']).reset_index(drop=True)
    val_y = val_y.sort_values(['site_id', 'forecast_year']).reset_index(drop=True)

    return X, val_X, test_X, y, val_y


res = get_global_data()
with open(os.path.join('..', 'assets', 'data', 'global_data_for_lstm_debugging.pkl'), 'wb') as f:
    pickle.dump(res, f)
