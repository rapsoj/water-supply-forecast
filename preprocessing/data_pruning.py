import os

import pandas as pd
from sklearn.decomposition import PCA

from consts import FIRST_FULL_GT_YEAR, JULY


def data_pruning(processed_data, ground_truth, FEAT_CORR_THRESH: float = .2):
    df = processed_data[(processed_data.forecast_year >= FIRST_FULL_GT_YEAR)
                        & (processed_data.date.dt.month <= JULY)
                        & ~(processed_data.forecast_year.isin(range(2005, 2024, 2)))].reset_index(drop=True)

    gt_col = list(set((ground_truth.site_id + ground_truth.forecast_year.astype(str))))
    df = df[(df.site_id + df.forecast_year.astype(str)).isin(gt_col)]

    # todo delete this?
    ft_col = list(set((df.site_id + df.forecast_year.astype(str))))
    ground_truth = ground_truth[(ground_truth.site_id + ground_truth.forecast_year.astype(str)).isin(ft_col)]
    ground_truth = ground_truth.sort_values(by=['site_id', 'forecast_year']).reset_index(drop=True)

    gt_map = ground_truth.drop_duplicates().set_index(['site_id', 'forecast_year']).volume.to_dict()
    ground_truth_vol = df.apply(lambda x: gt_map[(x.site_id, x.forecast_year)], axis='columns').reset_index(drop=True)
    # todo do this in a non hacky way
    ground_truth = pd.DataFrame(
        {'volume': ground_truth_vol.reset_index(drop=True), 'site_id': df.site_id.reset_index(drop=True),
         'forecast_year': df.forecast_year.reset_index(drop=True)})

    df = df.reset_index(drop=True)
    assert (df.site_id == ground_truth.site_id).all(), 'Site ids not matching in pruning'
    assert (df.forecast_year == ground_truth.forecast_year).all(), 'Forecast years not matching in pruning'
    assert (df.date.dt.year == ground_truth.forecast_year).all(), 'Forecast years and dates not matching in pruning'

    df['gt'] = ground_truth.volume

    cols_to_keep = ['site_id', 'date', 'forecast_year', 'time']

    df.drop(cols_to_keep, axis=1, inplace=True)
    corr_matrix = df.corr()

    cols_to_keep = list((set(corr_matrix.index[corr_matrix['gt'].abs() >= FEAT_CORR_THRESH]) | set(cols_to_keep))
                        - set(['gt']))

    rel_processed = processed_data[(processed_data.site_id + processed_data.forecast_year.astype(str)).isin(gt_col)]
    return rel_processed[cols_to_keep], ground_truth


def data_pca(processed_data, output_file_path: str = 'pruned_data', load_from_cache=True, n_components: int = 30):
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
