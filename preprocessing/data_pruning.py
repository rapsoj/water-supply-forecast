import os
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
from consts import FIRST_FULL_GT_YEAR, JULY, LAST_YEAR

# todo figure how to input the pruning threshold
def data_pruning(processed_data, ground_truth, pruning_threshold: float = 0.3):

    df = processed_data[(processed_data.forecast_year >= FIRST_FULL_GT_YEAR)
                                    & (processed_data.date.dt.month <= JULY)
                                    & ~(processed_data.forecast_year.isin(range(2005,2024,2)))].reset_index(drop=True)

    ground_truth = ground_truth.sort_values(by=['site_id', 'forecast_year']).reset_index(drop=True)

    assert (df.site_id == ground_truth.site_id).all(), 'Site ids not matching in pruning'
    assert (df.forecast_year == ground_truth.forecast_year).all(), 'Forecast years not matching in pruning'
    assert (df.date.dt.year == ground_truth.forecast_year).all(), 'Forecast years and dates not matching in pruning'

    df['gt'] = ground_truth.volume

    drop_cols = ['site_id', 'date', 'forecast_year']

    df.drop(drop_cols, axis=1, inplace=True)
    corr_matrix = df.corr()

    indices = list((set(corr_matrix.index[corr_matrix['gt'].abs() >= pruning_threshold]) | set(drop_cols)) - set(['gt']))

    pruned_df = processed_data[indices]
    return pruned_df


def data_pca(processed_data, output_file_path: str = 'pruned_data', load_from_cache = True, n_components: int = 30):
    if load_from_cache and os.path.exists(output_file_path):
        return pd.read_csv(output_file_path, parse_dates=['date'])

    pca = PCA(n_components)
    drop_cols = ['date', 'forecast_year', 'site_id']
    data_pca = processed_data.drop(columns=drop_cols)
    pca.fit(data_pca)
    data_pca = pca.transform(data_pca)
    for col in drop_cols:
        data_pca[col] = processed_data[col]
    return data_pca

