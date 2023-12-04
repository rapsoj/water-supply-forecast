import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.utils.fixes import parse_version, sp_version

from benchmark.benchmark_results import average_quantile_loss
from consts import DEF_QUANTILES, JULY


class StreamflowModel:
    def __init__(self, model, adapter=lambda x: x):
        self.model = model
        self.adapter = adapter
        self._loss = average_quantile_loss if isinstance(self.model, dict) else mean_squared_error

    def __call__(self, X: pd.DataFrame, adapt_feats: bool = True):
        if adapt_feats:
            X = self.adapter(X)

        assert (X.dtypes == float).all(), 'Error - wrong dtypes!'

        if isinstance(self.model, dict):
            pred = pd.DataFrame({q: q_model.predict(X) for q, q_model in self.model.items()})
        else:
            pred = pd.Series(self.model.predict(X))

        return pred

    def loss(self, X, y, adapt_feats: bool = True):
        pred = self(X, adapt_feats=adapt_feats)
        assert pred.shape[0] == y.shape[0], 'Error - predictions/ground truth mismatch!'

        loss = self._loss(y, pred)
        assert loss is not None and loss is not np.nan, 'Error - loss is None!'
        return loss


def general_pcr_fitter(X, y, val_X, val_y, quantile: bool = True, MAX_N_PCS: int = 30):
    pcr_X = pcr_adapt_features(X)
    pcr_val_X = pcr_adapt_features(val_X)

    MAX_N_PCS = min(MAX_N_PCS, *pcr_X.shape)

    min_v_loss = np.inf
    best_model = None
    for pc in range(1, MAX_N_PCS):
        model = pcr_fitter(pcr_X, y, pc=pc, quantile=quantile)

        loss = model.loss(pcr_val_X, val_y, adapt_feats=False)
        if min_v_loss > loss:
            min_v_loss = loss
            best_model = model

    return best_model


def pcr_adapt_features(X):
    return X[X.date.dt.month <= JULY].drop(columns=['date', 'forecast_year'])


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

        predictor = StreamflowModel(model, adapter=pcr_adapt_features)
    else:
        regressors = {}
        for q in DEF_QUANTILES:
            qregr = linear_model.QuantileRegressor(quantile=q, alpha=0.00, solver=solver)
            model = Pipeline([('pca', pca), ('quantile_regression', qregr)])
            model.fit(X, y)

            regressors[q] = model

        predictor = StreamflowModel(regressors, adapter=pcr_adapt_features)

    return predictor
