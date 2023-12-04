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


def general_pcr_fitter(X, y, val_X, val_y, test_X, quantile: bool = True, max_n_pcs: int = 29):
    pcr_X, pcr_val_X, pcr_test_X, test_dates = adapt_features(X, val_X, test_X)

    min_v_loss = np.inf
    best_model = None
    for pc in range(1, max_n_pcs):
        model = pcr_fitter(pcr_X, y, pc=pc, quantile=quantile)

        loss = model.loss(pcr_val_X, val_y)
        if min_v_loss > loss:
            min_v_loss = loss
            best_model = model

    return best_model, pcr_X, pcr_val_X, pcr_test_X, test_dates


def adapt_features(X, val_X, test_X):
    JULY = 7
    train_mask = X.date.dt.month <= JULY
    val_mask = val_X.date.dt.month <= JULY
    test_mask = test_X.date.dt.month <= JULY

    pcr_X = X[train_mask].drop(columns=['date', 'forecast_year'])
    pcr_val_X = val_X[val_mask].drop(columns=['date', 'forecast_year'])
    pcr_test_X = test_X[test_mask]
    test_dates = pcr_test_X.date.reset_index(drop=True)
    pcr_test_X = pcr_test_X.drop(columns=['date', 'forecast_year'])
    return pcr_X, pcr_val_X, pcr_test_X, test_dates


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
