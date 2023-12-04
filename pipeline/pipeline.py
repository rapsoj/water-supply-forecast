import numpy as np
import pandas as pd
import os

from benchmark.benchmark_results import benchmark_results, cache_preds
from models.fit_to_data import gen_basin_preds
from preprocessing.generic_preprocessing import get_processed_dataset
from preprocessing.pre_ml_processing import ml_preprocess_data

path = os.getcwd()

def run_pipeline(test_years: tuple = tuple(np.arange(2005, 2024, 2)),
                 validation_years: tuple = tuple(np.arange(1986, 2023, 8)), gt_col: str = 'volume',
                 load_from_cache: bool = True):
    print('Loading data')
    # todo add output_csv paths to preprocessing, especially the ml preprocessing
    basic_preprocessed_df = get_processed_dataset(load_from_cache=load_from_cache)

    # todo add explicit forecasting functionality, split train/test for forecasting earlier.
    #  currently everything is processed together. unsure if necessary
    processed_data, processed_ground_truth = ml_preprocess_data(basic_preprocessed_df, load_from_cache=load_from_cache)

    loaded_ground_truth = load_ground_truth(num_predictions=28)
    # Get training, validation and test sets
    train_features, val_features, test_features, train_gt, val_gt, test_gt = \
        train_val_test_split(processed_data, loaded_ground_truth, test_years, validation_years)

    # todo implement global models
    site_ids = processed_data.site_id.unique()
    # todo add the date in as a feature of some sort (perhaps just the month?, or datetime to int mod year (might already exist built-in)))
    non_feat_cols = ['site_id']
    final_submission_df = pd.DataFrame()
    for site_id in site_ids:
        print(f'Fitting to site {site_id}')
        train_site = train_features[train_features.site_id == site_id]
        # set-list in case non feature columns have NaNs
        drop_cols = list(set(non_feat_cols + train_site.columns[train_site.isna().any()].to_list()))

        train_site = train_site.drop(columns=drop_cols)
        train_site_gt = train_gt[train_gt.site_id == site_id]
        val_site = val_features[val_features.site_id == site_id].drop(columns=drop_cols)
        val_site_gt = val_gt[val_gt.site_id == site_id]
        test_site = test_features[test_features.site_id == site_id]
        test_dates = test_site.date
        test_site = test_site.drop(columns=drop_cols, errors='ignore')

        # todo deal with this using a global model
        if train_site_gt.empty or val_site_gt.empty:
            print(f'No ground truth data for site {site_id}')
            continue

        #train_site_gt, val_site_gt = ground_truth(train_site_gt, val_site_gt, num_predictions=28)

        train_pred, val_pred, test_pred, test_dates = gen_basin_preds(train_site, train_site_gt[gt_col], val_site,
                                                          val_site_gt[gt_col], test_site)

        results_id = f'{site_id}'
        print(f'Benchmarking results for site {site_id}')
        train_pred, val_pred, test_pred = benchmark_results(train_pred, train_site_gt[gt_col], val_pred,
                                                            val_site_gt[gt_col], test_pred, benchmark_id=results_id)

        site_submission = cache_preds(pred=test_pred, cache_id=results_id, site_id=site_id, pred_dates=test_dates)
        # todo figure out if there is a better way in which we wish to do this
        # Remove July values for detroit lake
        if site_id == "detroit_lake_inflow":
            mask = ~(site_submission.issue_date.dt.month == 7)
            site_submission = site_submission[mask]

        final_submission_df = pd.concat([final_submission_df, site_submission])

    # Get the correct order, sort in the way competition wants it
    ordered_site_ids = train_gt.site_id.unique()
    final_submission_df.site_id = final_submission_df.site_id.astype("category")
    final_submission_df.site_id = final_submission_df.site_id.cat.set_categories(ordered_site_ids)
    #final_submission_df = final_submission_df.sort_values(["site_id", "issue_date"])
    final_submission_df = final_submission_df.groupby(final_submission_df.issue_date.dt.year).apply(lambda x: x.sort_values(['site_id', 'issue_date']))

    final_submission_df.to_csv('final_pred.csv', index=False)

# To do implement this function to generate the ground truth
def load_ground_truth(num_predictions: int):
    ground_truth_df = pd.read_csv(os.path.join("..", "assets/data/", "train.csv"))
    # todo improve how we retrieve data for different sites, retrieving as much data as we can for each
    year_mask = (ground_truth_df.year >= 1986)
    ground_truth_df = ground_truth_df[year_mask].reset_index(drop=True)
    ground_truth_df = ground_truth_df.loc[ground_truth_df.index.repeat(num_predictions)]
    ground_truth_df['forecast_year'] = ground_truth_df.year
    ground_truth_df = ground_truth_df.drop(columns=['year'])
    return ground_truth_df
def ground_truth(train_gt: pd.DataFrame, val_gt: pd.DataFrame, num_predictions: int):
    # take "raw" train and validation gt dfs and sum over them seasonally and connect to the corresponding feature rows
    # (just multiply b a given number, because the labels are all the same atm)
    pcr_train_gt = pd.DataFrame()
    pcr_val_gt = pd.DataFrame()

    # todo fix ground truth data generation
    seasonal_mask_train = (train_gt.date.dt.month >= 4) & (train_gt.date.dt.month <= 8)
    seasonal_train = train_gt[seasonal_mask_train]
    pcr_train_gt = seasonal_train.groupby('forecast_year', as_index=False) \
        .volume.sum().reset_index(drop=True) # (Temporary measure) To offset 16 interpolated weeks on 4 months, approximate 4x multiplier

    pcr_train_gt = pcr_train_gt.loc[pcr_train_gt.index.repeat(num_predictions)].reset_index(drop=True)

    seasonal_mask_val = (val_gt.date.dt.month >= 4) & (val_gt.date.dt.month <= 8)
    seasonal_val = val_gt[seasonal_mask_val]
    pcr_val_gt = seasonal_val.groupby('forecast_year', as_index=False) \
        .volume.sum() # (Temporary measure) To offset the 16 interpolated weeks on 4 months, approximate 4x multiplier

    pcr_val_gt = pcr_val_gt.loc[pcr_val_gt.index.repeat(num_predictions)]

    # To do: Implement this completely, multiply the rows to the appropriate number and return

    return pcr_train_gt, pcr_val_gt

def train_val_test_split(feature_df: pd.DataFrame, gt_df: pd.DataFrame, test_years: tuple, validation_years: tuple):
    feature_df = feature_df.copy()
    gt_df = gt_df.copy()

    test_feature_mask = feature_df.forecast_year.isin(test_years)
    test_gt_mask = gt_df.forecast_year.isin(test_years)

    # todo check if dropping volume here is okay (should be ok I think because this volume is only used as features, not labels, and so gradually "unlocking" it over time should be alright, I think, although maybe this data is not available at test time)
    test_feature_df = feature_df[test_feature_mask].reset_index(drop=True)
    test_gt_df = gt_df[test_gt_mask].reset_index(drop=True)

    validation_feature_mask = feature_df.forecast_year.isin(validation_years)
    validation_gt_mask = gt_df.forecast_year.isin(validation_years)

    val_feature_df = feature_df[validation_feature_mask].reset_index(drop=True)
    val_gt_df = gt_df[validation_gt_mask].reset_index(drop=True)

    # todo decide where to input the start date 1983/1986 year
    train_feature_mask = ~validation_feature_mask & ~test_feature_mask & (feature_df.forecast_year >= 1986)
    train_gt_mask = ~validation_gt_mask & ~test_gt_mask & (gt_df.forecast_year >= 1986)

    train_feature_df = feature_df[train_feature_mask]
    train_gt_df = gt_df[train_gt_mask]

    assert train_feature_df.date.isin(val_feature_df.date).sum() == 0 and \
           train_feature_df.date.isin(test_feature_df.date).sum() == 0 and \
           val_feature_df.date.isin(test_feature_df.date).sum() == 0, \
        "Dates are overlapping between train, val, and test sets"

    # todo figure out why some things are empty here, e.g. test_gt_df
    return train_feature_df, val_feature_df, test_feature_df, train_gt_df, val_gt_df, test_gt_df


if __name__ == '__main__':
    run_pipeline()
