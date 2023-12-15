## Read in libraries
import os

import numpy as np
import pandas as pd

from preprocessing.helper_functions import cleaning, scaling, merge

## Import datasets
current_dir = os.getcwd()


def get_processed_dataset(output_file_path: str = 'transformed_vars.csv',
                          load_from_cache: bool = False) -> pd.DataFrame:
    if load_from_cache and os.path.exists(output_file_path):
        return pd.read_csv(output_file_path)

    # Importing steps
    df_mjo = cleaning.import_mjo(current_dir)
    df_nino = cleaning.import_nino(current_dir)
    df_oni = cleaning.import_oni(current_dir)
    df_pdo = cleaning.import_pdo(current_dir)
    df_pna = cleaning.import_pna(current_dir)
    df_soi1 = cleaning.import_soi1(current_dir)
    df_soi2 = cleaning.import_soi2(current_dir)
    df_flow = cleaning.import_flow(current_dir)
    df_grace = cleaning.import_grace(current_dir)
    #df_snotel = cleaning.import_snotel(current_dir)
    df_swann = cleaning.import_swann(current_dir)
    #df_cpc_prec = cleaning.import_cpc_prec(current_dir)
    #df_cpc_temp = cleaning.import_cpc_temp(current_dir)
    df_dem = cleaning.import_dem(current_dir)

    ## Pre-merge cleaning steps

    # Cleaning at this stage is only adjustments required to allow merging,
    # More serious cleaning follows in later stages.

    # Import dictionaries used to transform month variable to numeric

    # Run functions
    df_mjo = cleaning.clean_mjo(df_mjo)
    df_nino = cleaning.clean_nino(df_nino)
    df_oni = cleaning.clean_oni(df_oni)
    df_pdo = cleaning.clean_pdo(df_pdo)
    df_pna = cleaning.clean_pna(df_pna)
    df_soi1 = cleaning.clean_soi1(df_soi1)
    df_soi2 = cleaning.clean_soi2(df_soi2)
    df_flow = cleaning.clean_flow(df_flow)
    df_grace = cleaning.clean_grace(df_grace)
    #df_snotel = cleaning.clean_snotel(df_snotel)
    #df_cpc_prec = cleaning.clean_cpc_prec(df_cpc_prec)
    #df_cpc_temp = cleaning.clean_cpc_temp(df_cpc_temp)
    df_dem = cleaning.clean_dem(df_dem)
    df_swann = cleaning.clean_swann(df_swann)


    # Merging dataframes
    # Merge on site id, day, LEAD column
    #dataframes = [df_cpc_prec, df_cpc_temp]
    #df_merged = merge.merge_site_id_day_lead(dataframes)

    # Merge on site id, day
    dataframes = [df_grace, df_swann, df_flow]
    df_flow['day'] = np.nan  # input nan when we don't know which day of the month data was measured
    df_merged = merge.merge_site_id_day(dataframes)



    # Merge on day
    dataframes = [df_mjo, df_pna, df_soi1, df_soi2, df_pdo, df_nino, df_oni, df_merged]
    for df in dataframes:
        if 'day' in df.columns:
            continue
        df['day'] = np.nan
    df_merged = merge.merge_day(dataframes)

    # Merge on site_id
    df_merged = merge.merge_site_id([df_merged, df_dem])
    # Outlier cleaning todo

    # Imputation todo

    # Feature engineering from features.py todo

    # mark unknown measurement days explicitly
    df_merged.day[df_merged.day.isna()] = -1

    # Perform preprocessing on columns except day, month, year
    trans_vars = scaling.preprocess_dataframe(df_merged)

    # Output the DataFrame to a CSV file
    trans_vars.to_csv(output_file_path, index=False)

    return trans_vars


if __name__ == '__main__':
    get_processed_dataset()
