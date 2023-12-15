
from preprocessing.generic_preprocessing import get_processed_dataset
from preprocessing.pre_ml_processing import ml_preprocess_data
from consts import FIRST_FULL_GT_YEAR
from models.models import base_feature_adapter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def analyze_data():
    basic_preprocessed_df = get_processed_dataset(load_from_cache=True)

    # todo add explicit forecasting functionality, split train/test for forecasting earlier.
    #  currently everything is processed together. unsure if necessary
    processed_data = ml_preprocess_data(basic_preprocessed_df, load_from_cache=True)
    site_id = "animas_r_at_durango"
    processed_data = processed_data[processed_data.site_id == site_id].drop(columns=['site_id'])
    test_years = range(2005, 2024, 2)

    # This is across all training + validation years
    processed_data = processed_data[(processed_data.forecast_year >= FIRST_FULL_GT_YEAR) & (processed_data.date.dt.month <= 7) & ~(processed_data.date.dt.year.isin(test_years))]

    processed_data = base_feature_adapter(processed_data)


    n_components = 5
    pca = PCA(n_components=n_components)
    pca.fit(processed_data)
    pcs = sum([[idx for idx, val in enumerate(c) if abs(val) > 0.1] for c in pca.components_.T], [])
    pc_dict = {processed_data.columns[val]: pcs.count(val) for val in pcs}
    plt.bar(pc_dict.keys(), pc_dict.values())

    plt.show()

if __name__ == '__main__':
    analyze_data()


