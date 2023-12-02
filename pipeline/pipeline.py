from preprocessing.generic_preprocessing import get_processed_dataset


def run_pipeline():
    # todo add output_csv paths to preprocessing, especially the ml preprocessing
    basic_preprocessed_df = get_processed_dataset()

    test_years = NotImplemented
    for test_year in test_years:
        train, val, test = ml_preprocess_data(basic_preprocessed_df, test_year)

        # todo implement global models
        site_ids = NotImplemented
        for site_id in site_ids:
            train_site = train[train.site_id == site_id]
            val_site = val[val.site_id == site_id]
            test_site = test[test.site_id == site_id]

            train_gt = train_site.gt  # todo change to correct column name
            train_site = train_site.drop(columns='gt')
            train_pred, val_pred, test_pred = fit_to_data(train_site, val_site, test_site)

            benchmark_results(train_pred, train_gt, val_pred, val_site.gt, test_pred,
                              benchmark_id=f'{test_year}_{site_id}')


if __name__ == '__main__':
    run_pipeline()
