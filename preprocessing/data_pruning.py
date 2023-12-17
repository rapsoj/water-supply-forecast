import os
import pandas as pd
from sklearn.decomposition import PCA

def data_pruning(processed_data, output_file_path: str = 'pruned_data', load_from_cache = True, n_components: int = 30):
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

