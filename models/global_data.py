import os
import pickle

import numpy as np

from consts import N_PRED_MONTHS, N_PREDS_PER_MONTH, FIRST_FULL_GT_YEAR, N_PREDS
from pipeline.pipeline import load_ground_truth, train_val_test_split, extract_n_sites, \
    get_processed_data_and_ground_truth
from preprocessing.generic_preprocessing import get_processed_dataset
from preprocessing.pre_ml_processing import ml_preprocess_data
from preprocessing.data_pruning import prune_data


def get_global_data():
    load_from_cache = False
    use_additional_sites = True
    yearwise_validation = False


    n_sites = 50

    start_year = FIRST_FULL_GT_YEAR
    validation_years: tuple = tuple(np.arange(FIRST_FULL_GT_YEAR, 2023, 8))
    test_years: tuple = tuple(np.arange(2005, 2024, 2))

    processed_data, ground_truth = get_processed_data_and_ground_truth(load_from_cache=load_from_cache,
                                                                       use_additional_sites=use_additional_sites,
                                                                       test_years=test_years)

    processed_data, ground_truth = extract_n_sites(processed_data, ground_truth, n_sites)

    pruned_data = prune_data(processed_data, ground_truth)

    # Get training, validation and test sets
    X, val_X, test_X, y, val_y = \
        train_val_test_split(pruned_data, ground_truth, validation_years, validation_sites=validation_sites,
                             start_year=start_year, yearwise_validation=yearwise_validation)

    y = y.sort_values(['site_id', 'forecast_year']).reset_index(drop=True)
    val_y = val_y.sort_values(['site_id', 'forecast_year']).reset_index(drop=True)

    return X, val_X, test_X, y, val_y


res = get_global_data()
with open(os.path.join('..', 'assets', 'data', 'global_data_for_lstm_debugging_50sites.pkl'), 'wb') as f:
    pickle.dump(res, f)
