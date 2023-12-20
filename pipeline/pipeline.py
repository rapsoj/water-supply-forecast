import sys

# for when running in github codespaces, shouldn't affect anything otherwise
sys.path.append('/workspaces/water-supply-forecast')

import os
import random

import numpy as np
import pandas as pd
import torch

from benchmark.benchmark_results import benchmark_results, cache_preds, generate_submission_file, \
    cache_merged_submission_file
from consts import (JULY, FIRST_FULL_GT_YEAR, N_PREDS_PER_MONTH, N_PRED_MONTHS, DEBUG_N_SITES,
                    MIN_N_SITES, CORE_SITES, TEST_YEARS, ORDERED_SITE_IDS)
from models.fit_to_data import Ensemble_Type
from models.fit_to_data import ensemble_models
from models.fitters import general_pcr_fitter, general_xgboost_fitter, lstm_fitter
from preprocessing.generic_preprocessing import get_processed_dataset
from preprocessing.pre_ml_processing import ml_preprocess_data
from preprocessing.data_pruning import prune_data

path = os.getcwd()


def extract_n_sites(data: pd.DataFrame, ground_truth: pd.DataFrame, n_sites: int):
    assert n_sites >= MIN_N_SITES, f'Number of sites must be at least {MIN_N_SITES}'
    site_ids = data.site_id.unique()
    keeping_sites = CORE_SITES + tuple(set(site_ids) - set(CORE_SITES))[:n_sites - MIN_N_SITES]

    data = data[data.site_id.isin(keeping_sites)]
    ground_truth = ground_truth[ground_truth.site_id.isin(keeping_sites)]

    return data, ground_truth


def run_pipeline(validation_years: tuple = tuple(np.arange(FIRST_FULL_GT_YEAR, 2023, 8)),
                 validation_sites: tuple = tuple(ORDERED_SITE_IDS),
                 gt_col: str = 'volume',
                 load_from_cache: bool = True, start_year=FIRST_FULL_GT_YEAR, using_pca=False,
                 use_additional_sites: bool = True, n_sites: int = DEBUG_N_SITES, yearwise_validation=False):
    np.random.seed(0)
    random.seed(0)
    torch.random.manual_seed(0)

    print('Loading data')
    processed_data, ground_truth = get_processed_data_and_ground_truth(load_from_cache=load_from_cache,
                                                                       use_additional_sites=use_additional_sites)
    print('Extracting sites')
    processed_data, ground_truth = extract_n_sites(processed_data, ground_truth, n_sites)

    pruned_data = prune_data(processed_data, ground_truth)

    # Get training, validation and test sets
    train_features, val_features, test_features, train_gt, val_gt = \
        train_val_test_split(pruned_data, ground_truth, validation_years, validation_sites, start_year=start_year,
                             yearwise_validation=yearwise_validation)

    site_ids = pruned_data.site_id.unique()

    print('Running global models...')

    test_val_train_global_dfs = run_global_models(train_features, val_features, test_features, train_gt, val_gt, gt_col,
                                                  using_pca=using_pca, site_ids=site_ids)

    print('Running local models...')

    # test_val_train_local_dfs = run_local_models(train_features, val_features, test_features, train_gt, val_gt, gt_col, site_ids,
    #                                      using_pca=using_pca)

    # todo clean this up
    print('Ensembling global and local model submissions...')

    labels = ['pred', 'val', 'train']
    for idx, df in enumerate(test_val_train_global_dfs):
        label = labels[idx]

        full_dfs = df  # | global/local

        print(f'Dataset: {label}')

        final_df_dict = ensemble_models(full_dfs, 'final', ensemble_type=Ensemble_Type.BEST_PREDICTION)
        final_df = final_df_dict['final']
        cache_merged_submission_file(final_df, label)


def log_values(x: pd.Series):
    log_x = np.log(x)
    log_mean = np.mean(log_x)
    log_std = np.std(log_x)
    return log_x, log_mean, log_std


def scale_data(inp, mean, std):
    return (inp - mean) / std


def inv_scale_data(pred, mean, std):
    return pred * std + mean


def run_local_models(train_features, val_features, test_features, train_gt, val_gt, gt_col, site_ids, using_pca,
                     fitters=(general_xgboost_fitter,),
                     ):
    non_feat_cols = ['site_id']
    test_dfs = {}
    val_dfs = {}
    train_dfs = {}

    site_id_sets = []

    # get sitewise data
    for site_id in site_ids:
        train_site = train_features[train_features.site_id == site_id]
        # set-list in case non feature columns have NaNs
        drop_cols = list(set(non_feat_cols + train_site.columns[train_site.isna().any()].to_list()))

        train_site = train_site.drop(columns=drop_cols).reset_index(drop=True)
        train_site_gt = train_gt[train_gt.site_id == site_id].reset_index(drop=True)
        gt_std, gt_mean = train_site_gt[gt_col].std(), train_site_gt[gt_col].mean()

        val_site = val_features[val_features.site_id == site_id].drop(columns=drop_cols).reset_index(drop=True)
        val_site_gt = val_gt[val_gt.site_id == site_id].reset_index(drop=True)
        test_site = test_features[test_features.site_id == site_id].reset_index(drop=True)
        test_site = test_site.drop(columns=drop_cols, errors='ignore')


        train_site_gt[gt_col] = scale_data(train_site_gt[gt_col], gt_mean, gt_std)
        val_site_gt[gt_col] = scale_data(val_site_gt[gt_col], gt_mean, gt_std)

        # todo create namedtuple/smthing to make this prettier
        site_id_sets.append([train_site, train_site_gt, val_site, val_site_gt, test_site, gt_mean, gt_std])

    for fitter in fitters:
        for idx, site_id in enumerate(site_ids):
            print(f'Fitting to site {site_id}')
            train_site = site_id_sets[idx][0].copy()
            train_site_gt = site_id_sets[idx][1].copy()
            val_site = site_id_sets[idx][2].copy()
            val_site_gt = site_id_sets[idx][3].copy()
            test_site = site_id_sets[idx][4].copy()
            gt_mean = site_id_sets[idx][5]
            gt_std = site_id_sets[idx][6]

            if train_site_gt.empty or val_site_gt.empty:
                print(f'No ground truth data for site {site_id}!')
                continue

            hyper_tuned_model, model = fitter(train_site, train_site_gt[gt_col], val_site,
                                              val_site_gt[gt_col], using_pca=using_pca)
            train_pred = model(train_site)
            val_pred = hyper_tuned_model(val_site)
            test_pred = model(test_site)

            test_mask = test_site.date.dt.month <= JULY  # todo fix moving these test dates
            test_vals = test_site[test_mask]
            test_dates = test_vals.date.reset_index(drop=True)

            val_mask = val_features.date.dt.month <= JULY
            val_vals = val_features[val_mask]
            val_dates = val_vals.date.reset_index(drop=True).unique()
            train_mask = train_features.date.dt.month <= JULY
            train_vals = train_features[train_mask]
            train_dates = train_vals.date.reset_index(drop=True).unique()
            results_id = f'local_{fitter.__name__}_{site_id}'
            print(f'Benchmarking results for site {site_id}')

            # rescaling data
            train_pred = inv_scale_data(train_pred, gt_mean, gt_std)
            val_pred = inv_scale_data(val_pred, gt_mean, gt_std)
            test_pred = inv_scale_data(test_pred, gt_mean, gt_std)

            train_site_gt[gt_col] = inv_scale_data(train_site_gt[gt_col], gt_mean, gt_std)
            val_site_gt[gt_col] = inv_scale_data(val_site_gt[gt_col], gt_mean, gt_std)
            train_site_gt = train_site_gt.reset_index(drop=True)
            val_site_gt = val_site_gt.reset_index(drop=True)

            benchmark_results(train_pred, train_site_gt[gt_col], val_pred, val_site_gt[gt_col], benchmark_id=results_id)

            cache_preds(pred=test_pred, cache_id=results_id, site_id=site_id, pred_dates=test_dates, set_id='pred')
            cache_preds(pred=val_pred, cache_id=results_id, site_id=site_id, pred_dates=val_dates, set_id='val')
            cache_preds(pred=train_pred, cache_id=results_id, site_id=site_id, pred_dates=train_dates, set_id='train')

        print('Generating local model submission file...')
        test_df = generate_submission_file(ordered_site_ids=ORDERED_SITE_IDS, model_id='local',
                                           fitter_id=fitter.__name__, set_id='pred')
        val_df = generate_submission_file(ordered_site_ids=ORDERED_SITE_IDS, model_id='local',
                                          fitter_id=fitter.__name__, set_id='val')
        train_df = generate_submission_file(ordered_site_ids=ORDERED_SITE_IDS, model_id='local',
                                            fitter_id=fitter.__name__, set_id='train')

        test_dfs[f'local_{fitter.__name__}'] = test_df
        val_dfs[f'local_{fitter.__name__}'] = val_df
        train_dfs[f'local_{fitter.__name__}'] = train_df

    return test_dfs, val_dfs, train_dfs


def run_global_models(train_features, val_features, test_features, train_gt, val_gt, gt_col, site_ids, using_pca,
                      fitters=(lstm_fitter,), log_transform=False, yearwise_validation=False):
    train_features = train_features.sort_values(by=['site_id', 'date']).reset_index(
        drop=True)  # Might not be necessary (might already be that way) but just to make sure
    val_features = val_features.sort_values(by=['site_id', 'date']).reset_index(drop=True)
    train_gt = train_gt.sort_values(by=['site_id', 'forecast_year']).reset_index(drop=True)
    val_gt = val_gt.sort_values(by=['site_id', 'forecast_year']).reset_index(drop=True)

    # todo set up assert to check matching dates and site ids

    train_site_id_col = train_features.site_id.reset_index(drop=True)
    train_gt = train_gt.reset_index(drop=True)
    gt_std, gt_mean = train_gt[gt_col].std(), train_gt[gt_col].mean()

    val_site_id_col = val_features.site_id.reset_index(drop=True)
    val_gt = val_gt.reset_index(drop=True)
    test_site_id_col = test_features.site_id

    if log_transform:
        # todo make log transform work sitewise
        log_train_vals, log_mean, log_std = log_values(train_gt[gt_col])
        log_val_vals, _, _ = log_values(val_gt[gt_col])

        train_gt[gt_col] = scale_data(log_train_vals, log_mean, log_std)
        val_gt[gt_col] = scale_data(log_val_vals, log_mean, log_std)
    else:

        train_gt[gt_col] = scale_data(train_gt[gt_col], gt_mean, gt_std)
        val_gt[gt_col] = scale_data(val_gt[gt_col], gt_mean, gt_std)

    # todo perhaps find a better way of treating NaN values (Californian sites for volume + SNOTEL)
    train_features = train_features.fillna(0)
    val_features = val_features.fillna(0)
    test_features = test_features.fillna(0)
    test_dfs = {}
    val_dfs = {}
    train_dfs = {}

    for fitter in fitters:
        train_only_model, model = fitter(train_features, train_gt, val_features, val_gt)

        for site_id in CORE_SITES:
            results_id = f'global_{fitter.__name__}_{site_id}'
            train_site = train_features[train_site_id_col == site_id]
            train_site_gt = train_gt[train_gt.site_id == site_id]
            val_site = val_features[val_site_id_col == site_id]
            val_site_gt = val_gt[val_gt.site_id == site_id]
            test_site = test_features[test_site_id_col == site_id]

            # Get relevant dates
            test_mask = test_site.date.dt.month <= JULY
            test_vals = test_site[test_mask]
            test_dates = test_vals.date.reset_index(drop=True).unique()
            val_mask = val_site.date.dt.month <= JULY
            val_vals = val_site[val_mask]
            val_dates = val_vals.date.reset_index(drop=True).unique()
            train_mask = train_site.date.dt.month <= JULY
            train_vals = train_site[train_mask]
            train_dates = train_vals.date.reset_index(drop=True).unique()

            # todo fix this for sitewise validation tests
            train_pred = pd.DataFrame()
            val_pred = pd.DataFrame()
            test_pred = pd.DataFrame()

            if not train_site.empty:
                train_pred = model(train_site)
            if not val_site.empty:
                val_pred = train_only_model(val_site)
            if not test_site.empty:
                test_pred = model(test_site)

            # rescaling data
            if log_transform:
                if not train_site.empty:
                    train_pred = np.exp(inv_scale_data(train_pred, log_mean, log_std))
                    train_site_gt[gt_col] = np.exp(inv_scale_data(train_site_gt[gt_col], log_mean, log_std))
                if not val_site.empty:
                    val_pred = np.exp(inv_scale_data(val_pred, log_mean, log_std))
                    val_site_gt[gt_col] = np.exp(inv_scale_data(val_site_gt[gt_col], log_mean, log_std))
                if not test_site.empty:
                    test_pred = np.exp(inv_scale_data(test_pred, log_mean, log_std))
            else:
                if not train_site.empty:
                    train_pred = inv_scale_data(train_pred, gt_mean, gt_std)
                    train_site_gt[gt_col] = inv_scale_data(train_site_gt[gt_col], gt_mean, gt_std)
                if not val_site.empty:
                    val_pred = inv_scale_data(val_pred, gt_mean, gt_std)
                    val_site_gt[gt_col] = inv_scale_data(val_site_gt[gt_col], gt_mean, gt_std)
                if not test_site.empty:
                    test_pred = inv_scale_data(test_pred, gt_mean, gt_std)

            train_site_gt = train_site_gt.reset_index(drop=True)
            val_site_gt = val_site_gt.reset_index(drop=True)

            # todo make sitewise validation work, how should we track the validation losses?
            benchmark_results(train_pred=train_pred, train_gt=train_site_gt[gt_col], val_pred=val_pred,
                              val_gt=val_site_gt[gt_col], benchmark_id=results_id)
            if not test_pred.empty:
                cache_preds(pred=test_pred, cache_id=results_id, site_id=site_id, pred_dates=test_dates, set_id='pred')
            if not val_pred.empty:
                cache_preds(pred=val_pred, cache_id=results_id, site_id=site_id, pred_dates=val_dates, set_id='val')
            if not train_pred.empty:
                cache_preds(pred=train_pred, cache_id=results_id, site_id=site_id, pred_dates=train_dates,
                            set_id='train')

        print('Generating global model submission file...')
        df_test = generate_submission_file(ordered_site_ids=ORDERED_SITE_IDS, model_id='global',
                                           fitter_id=fitter.__name__,
                                           set_id='pred')
        if not yearwise_validation:
            # Here we generate files only with the sites which actually exist in the set, different from before when we were generating the files with only the old sites
            validation_sites = val_features.site_id.unique()
            df_val = generate_submission_file(ordered_site_ids=validation_sites, model_id='global',
                                              fitter_id=fitter.__name__,
                                              set_id='val')
            train_sites = train_features.site_id.unique()

            df_train = generate_submission_file(ordered_site_ids=train_sites, model_id='global',
                                                fitter_id=fitter.__name__,
                                                set_id='train')
        else:
            df_val = generate_submission_file(ordered_site_ids=ORDERED_SITE_IDS, model_id='global',
                                              fitter_id=fitter.__name__,
                                              set_id='val')
            df_train = generate_submission_file(ordered_site_ids=ORDERED_SITE_IDS, model_id='global',
                                                fitter_id=fitter.__name__,
                                                set_id='train')
        test_dfs[f'global_{fitter.__name__}'] = df_test
        val_dfs[f'global_{fitter.__name__}'] = df_val
        train_dfs[f'global_{fitter.__name__}'] = df_train

    return test_dfs, val_dfs, train_dfs


def load_ground_truth(num_predictions: int, additional_sites: bool = False) -> pd.DataFrame:
    ground_truth_df = pd.read_csv(os.path.join("..", "assets", "data", "train.csv"))

    if additional_sites:
        additional_ground_truth_df = pd.read_csv(os.path.join("..", "assets", "data", "additional_train.csv"))

        additional_ground_truth_df = additional_ground_truth_df[
            ~additional_ground_truth_df.site_id.isin(ORDERED_SITE_IDS)]
        ground_truth_df = pd.concat([ground_truth_df, additional_ground_truth_df])

    # todo improve how we retrieve data for different sites, retrieving as much data as we can for each
    year_mask = (ground_truth_df.year >= FIRST_FULL_GT_YEAR)
    ground_truth_df = ground_truth_df[year_mask].reset_index(drop=True)

    ground_truth_df = ground_truth_df.loc[ground_truth_df.index.repeat(num_predictions)]

    ground_truth_df['forecast_year'] = ground_truth_df.year

    ground_truth_df = ground_truth_df.drop(columns=['year'])

    return ground_truth_df


def train_val_test_split(feature_df: pd.DataFrame, gt_df: pd.DataFrame, validation_years: tuple,
                         validation_sites: tuple,
                         start_year: int = FIRST_FULL_GT_YEAR, yearwise_validation: bool = False):
    feature_df = feature_df.copy()
    gt_df = gt_df.copy()

    test_feature_mask = feature_df.forecast_year.isin(TEST_YEARS) & feature_df.site_id.isin(ORDERED_SITE_IDS)

    test_feature_df = feature_df[test_feature_mask].reset_index(drop=True)
    if yearwise_validation:
        val_mask = feature_df.forecast_year.isin(validation_years)
        val_gt_mask = gt_df.forecast_year.isin(validation_years)

        val_feature_df = feature_df[val_mask].reset_index(drop=True)
        val_gt_df = gt_df[val_gt_mask].reset_index(drop=True)
    else:
        val_mask = feature_df.site_id.isin(validation_sites) & ~feature_df.forecast_year.isin(TEST_YEARS)
        val_gt_mask = gt_df.site_id.isin(validation_sites) & ~gt_df.forecast_year.isin(TEST_YEARS)

        val_feature_df = feature_df[val_mask].reset_index(drop=True)
        val_gt_df = gt_df[val_gt_mask].reset_index(drop=True)

    # Specifically get the years here since you have the extra criterion of sites in the test_mask which can cause additional sites being added on test years to train features
    test_features_years_mask = feature_df.forecast_year.isin(TEST_YEARS)
    test_gt_years_mask = gt_df.forecast_year.isin(TEST_YEARS)
    train_mask = ~val_mask & ~test_features_years_mask & (feature_df.forecast_year >= start_year)
    train_gt_mask = ~val_gt_mask & ~test_gt_years_mask & (gt_df.forecast_year >= start_year)

    # todo filter this in a more dynamic way, getting the minimum year
    train_feature_df = feature_df[train_mask]
    train_gt_df = gt_df[train_gt_mask]
    if yearwise_validation:
        assert train_feature_df.date.isin(val_feature_df.date).sum() == 0 and \
               train_feature_df.date.isin(test_feature_df.date).sum() == 0 and \
               val_feature_df.date.isin(test_feature_df.date).sum() == 0, \
            "Dates are overlapping between train, val, and test sets"
    else:
        assert train_feature_df.date.isin(test_feature_df.date).sum() == 0 and \
               val_feature_df.date.isin(test_feature_df.date).sum() == 0, \
            "Dates are overlapping between train, val, and test sets"
        assert train_feature_df.site_id.isin(val_feature_df.site_id).sum() == 0, \
            "Site IDs are overlapping between train"

    assert test_feature_df.size > 0, "No test features"
    assert val_feature_df.size > 0, "No validation features"
    assert train_feature_df.size > 0, "No train features"
    # todo figure out why some things are empty here, e.g. test_gt_df
    return train_feature_df, val_feature_df, test_feature_df, train_gt_df, val_gt_df


def make_gt_and_features_siteyear_consistent(processed_data: pd.DataFrame, ground_truth: pd.DataFrame) -> \
        (pd.DataFrame, pd.DataFrame):
    # todo clean this
    df = processed_data[(processed_data.forecast_year >= FIRST_FULL_GT_YEAR)
                        & (processed_data.date.dt.month <= JULY)].reset_index(drop=True)

    gt_col = list(set((ground_truth.site_id + ground_truth.forecast_year.astype(str))))
    df = df[(df.site_id + df.forecast_year.astype(str)).isin(gt_col)].reset_index(drop=True)

    ft_col = list(set((df.site_id + df.forecast_year.astype(str))))
    ground_truth = ground_truth[(ground_truth.site_id + ground_truth.forecast_year.astype(str)).isin(ft_col)]
    ground_truth = ground_truth.sort_values(by=['site_id', 'forecast_year']).reset_index(drop=True)

    assert (df.site_id == ground_truth.site_id).all(), 'Site ids not matching in pruning'
    assert (df.forecast_year == ground_truth.forecast_year).all(), 'Forecast years not matching in pruning'
    assert (df.date.dt.year == ground_truth.forecast_year).all(), 'Forecast years and dates not matching in pruning'

    # That was all fun playing around with the params, let's retrieve the relevant data
    rel_processed_data = processed_data[(processed_data.site_id + processed_data.forecast_year.astype(str)).isin(gt_col)
                                        | (processed_data.forecast_year.isin(TEST_YEARS) &
                                           processed_data.site_id.isin(ground_truth.site_id.unique()))] \
        .reset_index(drop=True)

    return rel_processed_data, ground_truth


def get_processed_data_and_ground_truth(load_from_cache=True, use_additional_sites=True):
    basic_preprocessed_df = get_processed_dataset(load_from_cache=load_from_cache,
                                                  use_additional_sites=use_additional_sites)

    # todo add explicit forecasting functionality, split train/test for forecasting earlier.
    #  currently everything is processed together. unsure if necessary
    processed_data = ml_preprocess_data(basic_preprocessed_df, load_from_cache=load_from_cache,
                                        use_additional_sites=use_additional_sites)

    # Data sanity check
    assert all([data_type == float for data_type in processed_data
               .drop(columns=['date', 'forecast_year', 'site_id']).dtypes]), "All features are not floats"

    ground_truth = load_ground_truth(num_predictions=N_PRED_MONTHS * N_PREDS_PER_MONTH,
                                     additional_sites=use_additional_sites)

    processed_data, ground_truth = make_gt_and_features_siteyear_consistent(processed_data, ground_truth)

    assert processed_data.site_id.nunique() >= len(CORE_SITES), "Not enough sites in processed data"
    assert ground_truth.site_id.nunique() >= len(CORE_SITES), "Not enough sites in ground truth"

    return processed_data, ground_truth


if __name__ == "__main__":
    run_pipeline()
