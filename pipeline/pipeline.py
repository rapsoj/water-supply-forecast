# todo add output_csvs to preprocessing
def main():
    basic_preprocessed_df = get_processed_dataset()

    test_years = NotImplemented
    for test_year in test_years:
        train, test = ml_preprocess_data(basic_preprocessed_df, test_year)

    # todo implement global models
    site_ids = NotImplemented
    for site_id in site_ids:
        pass


if __name__ == '__main__':
    pass
