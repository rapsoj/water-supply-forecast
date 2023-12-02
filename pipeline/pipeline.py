from benchmark.benchmark_results import benchmark_results, cache_preds
from models.fit_to_data import gen_basin_preds
from preprocessing.generic_preprocessing import get_processed_dataset


def run_pipeline(gt_col='gt'):  # todo change to correct column name
    # todo add output_csv paths to preprocessing, especially the ml preprocessing
    basic_preprocessed_df = get_processed_dataset()

    test_years = NotImplemented
    for test_year in test_years:
        train, val, test = ml_preprocess_data(basic_preprocessed_df, test_year)
        assert gt_col not in test.columns, 'Error - test should not have a ground truth!'

        # todo implement global models
        site_ids = NotImplemented
        for site_id in site_ids:
            train_site = train[train.site_id == site_id]
            val_site = val[val.site_id == site_id]
            test_site = test[test.site_id == site_id]

            train_gt = train_site.gt
            train_site = train_site.drop(columns=gt_col)
            val_gt = val_site.gt
            val_site = val_site.drop(columns=gt_col)

            train_pred, val_pred, test_pred = gen_basin_preds(train_site, val_site, test_site)

            results_id = f'{test_year}_{site_id}'
            train_pred, val_pred, test_pred = benchmark_results(train_pred, train_gt, val_pred, val_gt, test_pred,
                                                                benchmark_id=results_id)
            # todo make sure that `dates` is the dates column
            cache_preds(pred=test_pred, cache_id=results_id, site_id=site_id, pred_dates=test_site.dates)


if __name__ == '__main__':
    run_pipeline()
