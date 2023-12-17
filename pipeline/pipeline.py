import os

import numpy as np
import pandas as pd

from benchmark.benchmark_results import benchmark_results, cache_preds, generate_submission_file, \
    cache_merged_submission_file
from consts import JULY, FIRST_FULL_GT_YEAR, N_PREDS_PER_MONTH, N_PRED_MONTHS
from models.fit_to_data import Ensemble_Type
from models.fit_to_data import ensemble_models
from models.models import general_pcr_fitter, general_xgboost_fitter
from preprocessing.generic_preprocessing import get_processed_dataset
from preprocessing.pre_ml_processing import ml_preprocess_data
from preprocessing.data_pruning import data_pruning

path = os.getcwd()

def run_pipeline(test_years: tuple = tuple(np.arange(2005, 2024, 2)),
                 validation_years: tuple = tuple(np.arange(FIRST_FULL_GT_YEAR, 2023, 8)), gt_col: str = 'volume',
                 load_from_cache: bool = False, start_year=FIRST_FULL_GT_YEAR, using_pca=False):
    print('Loading data')
    basic_preprocessed_df = get_processed_dataset(load_from_cache=load_from_cache)

    # todo add explicit forecasting functionality, split train/test for forecasting earlier.
    #  currently everything is processed together. unsure if necessary
    processed_data = ml_preprocess_data(basic_preprocessed_df, load_from_cache=load_from_cache)

    # Data sanity check
    # Check types (do we wish to also check that date, forecast_year and site_id are the correct types here?
    assert all([data_type == float for data_type in processed_data
               .drop(columns=['date', 'forecast_year', 'site_id']).dtypes]), "All features are not floats"
    # assert len(processed_data.site_id[processed_data.volume.isna()].unique()) == 3, \
    #    "More than 3 sites having NaNs in volume (should only be the California sites)"
    # assert len(processed_data.site_id[processed_data.SNWD_DAILY.isna()].unique()) == 1, \
    #    "More than 1 site has NaNs in SNWD_DAILY (should only be american river folsom)"

    ground_truth = load_ground_truth(num_predictions=N_PRED_MONTHS * N_PREDS_PER_MONTH)
    # Get training, validation and test sets
    train_features, val_features, test_features, train_gt, val_gt = \
        train_val_test_split(processed_data, ground_truth, test_years, validation_years, start_year=start_year)

    site_ids = processed_data.site_id.unique()

    print('Running global models...')

    test_val_train_global_dfs = run_global_models(train_features, val_features, test_features, \
                                  train_gt, val_gt, gt_col, site_ids)

    print('Running local models...')

    test_val_train_dfs = run_local_models(train_features, val_features, test_features, train_gt, val_gt, gt_col, site_ids,
                                          using_pca=using_pca)

    print('Ensembling global and local model submissions...')

    labels = ['pred', 'val', 'train']

    for idx, df in enumerate(test_val_train_dfs):

        label = labels[idx]

        full_dfs = df# | test_val_train_global_dfs[idx]

        final_df_dict = ensemble_models(full_dfs,'final', ensemble_type=Ensemble_Type.BEST_PREDICTION)
        final_df = final_df_dict['final']
        cache_merged_submission_file(final_df, label)


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

        ordered_site_ids = train_gt.site_id.drop_duplicates().tolist()
        print('Generating local model submission file...')
        test_df = generate_submission_file(ordered_site_ids=ordered_site_ids, model_id='local', fitter_id=fitter.__name__, set_id='pred')
        val_df = generate_submission_file(ordered_site_ids=ordered_site_ids, model_id='local', fitter_id=fitter.__name__, set_id='val')
        train_df = generate_submission_file(ordered_site_ids=ordered_site_ids, model_id='local', fitter_id=fitter.__name__, set_id='train')

        test_dfs[f'local_{fitter.__name__}'] = test_df
        val_dfs[f'local_{fitter.__name__}'] = val_df
        train_dfs[f'local_{fitter.__name__}'] = train_df

    return test_dfs, val_dfs, train_dfs


def run_global_models(train_features, val_features, test_features, train_gt, val_gt, gt_col, site_ids, using_pca,
                      fitters=(general_xgboost_fitter,)):
    drop_cols = ['site_id']
    train_site_id_col = train_features.site_id.reset_index(drop=True)
    train_features = train_features.drop(columns=drop_cols).reset_index(drop=True)
    train_gt = train_gt.reset_index(drop=True)
    gt_std, gt_mean = train_gt[gt_col].std(), train_gt[gt_col].mean()

    val_site_id_col = val_features.site_id.reset_index(drop=True)
    val_features = val_features.drop(columns=drop_cols).reset_index(drop=True)
    val_gt = val_gt.reset_index(drop=True)
    test_site_id_col = test_features.site_id
    test_features = test_features.drop(columns=drop_cols, errors='ignore')

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
        hyper_tuned_model, model = fitter(train_features, train_gt[gt_col], val_features, val_gt[gt_col], using_pca=using_pca)

        test_mask = test_features.date.dt.month <= JULY
        test_vals = test_features[test_mask]
        test_dates = test_vals.date.reset_index(drop=True).unique()
        val_mask = val_features.date.dt.month <= JULY
        val_vals = val_features[val_mask]
        val_dates = val_vals.date.reset_index(drop=True).unique()
        train_mask = train_features.date.dt.month <= JULY
        train_vals = train_features[train_mask]
        train_dates = train_vals.date.reset_index(drop=True).unique()


        for site_id in site_ids:
            results_id = f'global_{fitter.__name__}_{site_id}'
            train_site = train_features[train_site_id_col == site_id]
            train_site_gt = train_gt[train_gt.site_id == site_id]
            val_site = val_features[val_site_id_col == site_id]
            val_site_gt = val_gt[val_gt.site_id == site_id]

            test_site = test_features[test_site_id_col == site_id]
            train_pred = model(train_site)
            val_pred = hyper_tuned_model(val_site)
            test_pred = model(test_site)

            # rescaling data
            train_pred = inv_scale_data(train_pred, gt_mean, gt_std)
            val_pred = inv_scale_data(val_pred, gt_mean, gt_std)
            test_pred = inv_scale_data(test_pred, gt_mean, gt_std)

            train_site_gt[gt_col] = inv_scale_data(train_site_gt[gt_col], gt_mean, gt_std)
            val_site_gt[gt_col] = inv_scale_data(val_site_gt[gt_col], gt_mean, gt_std)

            train_site_gt = train_site_gt.reset_index(drop=True)
            val_site_gt = val_site_gt.reset_index(drop=True)

            benchmark_results(train_pred, train_site_gt[gt_col], val_pred,
                              val_site_gt[gt_col], val_pred, benchmark_id=results_id)

            cache_preds(pred=test_pred, cache_id=results_id, site_id=site_id, pred_dates=test_dates, set_id='pred')
            cache_preds(pred=val_pred, cache_id=results_id, site_id=site_id, pred_dates=val_dates, set_id='val')
            cache_preds(pred=train_pred, cache_id=results_id, site_id=site_id, pred_dates=train_dates, set_id='train')

        ordered_site_ids = train_gt.site_id.drop_duplicates().tolist()
        print('Generating global model submission file...')
        df_test = generate_submission_file(ordered_site_ids=ordered_site_ids, model_id='global', fitter_id=fitter.__name__,
                                      set_id='pred')
        df_val = generate_submission_file(ordered_site_ids=ordered_site_ids, model_id='global', fitter_id=fitter.__name__,
                                      set_id='val')
        df_train = generate_submission_file(ordered_site_ids=ordered_site_ids, model_id='global', fitter_id=fitter.__name__,
                                      set_id='train')

        test_dfs[f'global_{fitter.__name__}'] = df_test
        val_dfs[f'global_{fitter.__name__}'] = df_val
        train_dfs[f'global_{fitter.__name__}_train'] = df_train

    return test_dfs, val_dfs, train_dfs


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


if __name__ == '__main__':
    run_pipeline()
