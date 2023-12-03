import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils.fixes import parse_version, sp_version

from benchmark.benchmark_results import average_quantile_loss
from consts import DEF_QUANTILES


class StreamflowModel:
    def __init__(self, model):
        self.model = model
        self.loss = average_quantile_loss if isinstance(self.model, dict) else mean_squared_error

    def __call__(self, X):
        assert (X.dtypes == float).all(), 'Error - wrong dtypes!'

        if isinstance(self.model, dict):
            pred = pd.DataFrame({q: q_model(X) for q, q_model in self.model.items()})
        else:
            pred = pd.Series(self.model(X))

        return pred

    def loss(self, X, y):
        pred = self.model(X)
        return self.loss(y, pred)


def general_pcr_fitter(X, y, val_X, val_y, quantile: bool = True, max_n_pcs: int = 30):
    min_v_loss = np.inf
    best_model = None
    for pc in range(1, max_n_pcs):
        model = pcr_fitter(X, y, pc=pc, quantile=quantile)

        loss = model.loss(val_X, val_y)
        if min_v_loss < loss:
            min_v_loss = loss
            best_model = model

    return best_model


def pcr_fitter(X, y, pc, quantile: bool = True,
               solver="highs" if sp_version >= parse_version("1.6.0") else "inferior-point"):
    assert (X.dtypes == float).all(), 'Error - wrong dtypes!'

    # Instantiate the PCA object
    pca = PCA()

    # Run PCA
    Xreg = pca.fit_transform(X)[:, :pc]

    if not quantile:
        # Instantiate linear regression object
        regr = linear_model.LinearRegression()

        regr.fit(Xreg, y)

        predictor = StreamflowModel(regr)
    else:
        regressors = {}
        for q in DEF_QUANTILES:
            qregr = linear_model.QuantileRegressor(quantile=q, alpha=0.00, solver=solver)
            qregr.fit(Xreg, y)

            regressors[q] = qregr

        predictor = StreamflowModel(regressors)

    return predictor
