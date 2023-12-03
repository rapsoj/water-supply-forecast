import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.fixes import parse_version, sp_version

from benchmark.benchmark_results import average_quantile_loss
from consts import DEF_QUANTILES


class StreamflowModel:
    def __init__(self, model):
        self.model = model
        self._loss = average_quantile_loss if isinstance(self.model, dict) else mean_squared_error

    def __call__(self, X):
        assert (X.dtypes == float).all(), 'Error - wrong dtypes!'

        if isinstance(self.model, dict):
            pred = pd.DataFrame({q: q_model.predict(X) for q, q_model in self.model.items()})
        else:
            pred = pd.Series(self.model.predict(X))

        return pred

    def loss(self, X, y):
        pred = self(X)
        return self._loss(y, pred)


def general_pcr_fitter(X, y, val_X, val_y, quantile: bool = True, max_n_pcs: int = 30):
    pcr_train_gt, pcr_val_gt = pcr_ground_truth(y, val_y, 16)

    min_v_loss = np.inf
    best_model = None
    for pc in range(1, max_n_pcs):
        model = pcr_fitter(X, y, pc=pc, quantile=quantile)

        loss = model.loss(val_X, val_y)
        if min_v_loss > loss:
            min_v_loss = loss
            best_model = model

    return best_model

# To do implement this function to generate the ground truth
def pcr_ground_truth(train_gt: pd.DataFrame, val_gt: pd.DataFrame, num_predictions: int):
    # take "raw" train and validation gt dfs and sum over them seasonally and connect to the corresponding feature rows
    # (just multiply b a given number, because the labels are all the same atm)
    pcr_train_gt = pd.DataFrame()
    pcr_val_gt = pd.DataFrame()

    pcr_train_gt = train_gt.groupby('forecast_year', as_index=False) \
        .volume.sum()

    pcr_train_gt = pcr_train_gt.loc[pcr_train_gt.index.repeat(num_predictions)]

    pcr_val_gt = val_gt.groupby('forecast_year') \
        .volume.sum()

    pcr_val_gt = pcr_val_gt.loc[pcr_train_gt.index.repeat(num_predictions)]

    # To do: Implement this completely, multiply the rows to the appropriate number and return

    return pcr_val_gt, pcr_train_gt

def pcr_fitter(X, y, pc, quantile: bool = True,
               solver="highs" if sp_version >= parse_version("1.6.0") else "inferior-point"):
    assert (X.dtypes == float).all(), 'Error - wrong dtypes!'

    # Instantiate the PCA object
    pca = PCA(n_components=pc)

    if not quantile:
        # Instantiate linear regression object
        regr = linear_model.LinearRegression()
        model = Pipeline([('pca', pca), ('linear_regression', regr)])

        model.fit(X, y)

        predictor = StreamflowModel(model)
    else:
        regressors = {}
        for q in DEF_QUANTILES:
            qregr = linear_model.QuantileRegressor(quantile=q, alpha=0.00, solver=solver)
            model = Pipeline([('pca', pca), ('quantile_regression', qregr)])
            model.fit(X, y)

            regressors[q] = model

        predictor = StreamflowModel(regressors)

    return predictor
