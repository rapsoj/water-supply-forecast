import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.utils.fixes import parse_version, sp_version

from benchmark.benchmark_results import average_quantile_loss
from consts import DEF_QUANTILES, JULY


def base_feature_adapter(X):
    return X[X.date.dt.month <= JULY].drop(columns=['date', 'forecast_year'])


class StreamflowModel:
    def __init__(self, model, adapter=base_feature_adapter):
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


def xgboost_fitter(X, y, val_X, val_y, quantile: bool = True):
    xgb_X = base_feature_adapter(X)
    xgb_val_X = base_feature_adapter(val_X)
    combined_X = pd.concat([xgb_X, xgb_val_X])
    combined_y = pd.concat([y, val_y])

    if quantile:
        hyper_tuned_q_models = {}
        best_q_models = {}
        for q in DEF_QUANTILES:
            h_model = GradientBoostingRegressor(loss='quantile', alpha=q, max_depth=7)
            h_model.fit(xgb_X, y)
            hyper_tuned_q_models[q] = h_model
            b_model = GradientBoostingRegressor(loss='quantile', alpha=q, max_depth=7)
            b_model.fit(combined_X, combined_y)
            best_q_models[q] = b_model

        return StreamflowModel(hyper_tuned_q_models), StreamflowModel(best_q_models)

    model = GradientBoostingRegressor(loss='quantile', alpha=.5)
    model.fit(xgb_X, y)
    return StreamflowModel(model), StreamflowModel(model)

def k_nearest_neighbors_fitter(X, y, val_X, val_y):
    knn_X = base_feature_adapter(X)
    knn_val_X = base_feature_adapter(val_X)
    combined_X = pd.concat([knn_X, knn_val_X])
    combined_y = pd.concat([y, val_y])

    # todo implement k nearest neighbors


def general_pcr_fitter(X, y, val_X, val_y, quantile: bool = True, MAX_N_PCS: int = 50):
    pcr_X = base_feature_adapter(X)
    pcr_val_X = base_feature_adapter(val_X)

    MAX_N_PCS = min(MAX_N_PCS, *pcr_X.shape)

    min_v_loss = np.inf
    hyper_tuned_model = None
    optimal_pc = 0
    for pc in range(1, MAX_N_PCS):
        model = pcr_fitter(pcr_X, y, pc=pc, quantile=quantile)

        loss = model.loss(pcr_val_X, val_y, adapt_feats=False)
        if min_v_loss >= loss:
            optimal_pc = pc
            min_v_loss = loss
            hyper_tuned_model = model
    print(optimal_pc)
    # train hyperparameter tuned model on train+validation set
    combined_X = pd.concat([pcr_X, pcr_val_X])
    combined_y = pd.concat([y, val_y])

    best_model = pcr_fitter(combined_X, combined_y, pc=optimal_pc, quantile=quantile)
    return hyper_tuned_model, best_model


def pcr_fitter(X, y, pc, quantile: bool = True, solver="highs"):
    assert (X.dtypes == float).all(), 'Error - wrong dtypes!'

    # Instantiate the PCA object
    pca = PCA(n_components=pc)

    if not quantile:
        # Instantiate linear regression object
        regr = linear_model.LinearRegression()
        model = Pipeline([('pca', pca), ('linear_regression', regr)])

        model.fit(X, y)

        predictor = StreamflowModel(model, adapter=base_feature_adapter)
    else:
        regressors = {}
        for q in DEF_QUANTILES:
            qregr = linear_model.QuantileRegressor(quantile=q, solver=solver)
            model = Pipeline([('pca', pca), ('quantile_regression', qregr)])
            model.fit(X, y)

            regressors[q] = model

        predictor = StreamflowModel(regressors)

    return predictor
