import pandas as pd


def features_from_raw_data(raw_data: pd.DataFrame, fill_nan_means: pd.DataFrame, scaling_means: pd.DataFrame,
                           scaling_std: pd.DataFrame, group_cols: tuple) -> pd.DataFrame:

    # intput: df_merged, info: scaling dfs (fillna, mean and std), group_cols ('site_id, 'month')
    # First, scale in accordance with scale_dataframe

    scaled_data = scale_raw_data(raw_data, fill_nan_means, scaling_means, scaling_std, group_cols)

def scale_raw_data(raw_data: pd.DataFrame, fill_nan_means: pd.DataFrame, scaling_means: pd.DataFrame,
                           scaling_std: pd.DataFrame, group_cols: tuple) -> pd.DataFrame:

    filled_data = raw_data.groupby(group_cols).fillna(fill_nan_means)

    filled_data[non_fillna_cols] = filled_data[non_fillna_cols] * scaling_std + scaling_means

    raise NotImplementedError
