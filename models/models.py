import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_pinball_loss, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils.fixes import parse_version, sp_version

from benchmark.benchmark_results import average_quantile_loss
from consts import DEF_QUANTILES, JULY


def base_feature_adapter(X, pca=None):

    X = (X[X.date.dt.month <= JULY].drop(columns=['date', 'forecast_year']))
    return X


class StreamflowModel:
    def __init__(self, model, adapter=base_feature_adapter, pca=None):
        self.model = model
        self.adapter = adapter
        self._loss = average_quantile_loss if isinstance(self.model, dict) else mean_squared_error
        self.pca = pca

    def __call__(self, X: pd.DataFrame, adapt_feats: bool = True):
        if adapt_feats:
            X = self.adapter(X)

        assert (X.dtypes == float).all(), 'Error - wrong dtypes!'

        if self.pca is not None: # for non PCR models
            X = self.pca.transform(X)



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


def general_xgboost_fitter(X, y, val_X, val_y, MAX_N_PCS = 30, quantile: bool = True, using_pca=True):

    xgb_X = base_feature_adapter(X)
    xgb_val_X = base_feature_adapter(val_X)
    combined_X = pd.concat([xgb_X, xgb_val_X])
    combined_y = pd.concat([y, val_y])
    if using_pca:
        MAX_N_PCS = min(MAX_N_PCS, *xgb_X.shape)
        optimal_pc = 1
        min_v_loss = np.inf

        for pc in range(1, MAX_N_PCS):

            model = xgboost_fitter(xgb_X, y, pc=pc, quantile=quantile, using_pca=True)

            loss = model.loss(xgb_val_X, val_y, adapt_feats=False)
            if min_v_loss >= loss:
                optimal_pc = pc
                min_v_loss = loss
                hyper_tuned_model = model

        print(optimal_pc)

        best_model = xgboost_fitter(combined_X, combined_y, pc=optimal_pc, quantile=quantile)
        return hyper_tuned_model, best_model
    else:
        hyper_tuned_model, best_model = xgboost_fitter(xgb_X, y, xgb_val_X, val_y, using_pca=False)
        return hyper_tuned_model, best_model


def xgboost_fitter(xgb_X, y, xgb_val_X=None, val_y=None, quantile: bool = True, pc: int = None, using_pca=True):
    if using_pca:

        pca = PCA(n_components=pc)
        pca.fit(xgb_X)  # todo figure out whether to fit on val set to
        pca_xgb_X = pca.transform(xgb_X)

        if quantile:
            hyper_tuned_q_models = {}
            for q in DEF_QUANTILES:
                h_model = GradientBoostingRegressor(loss='quantile', alpha=q)
                h_model.fit(pca_xgb_X, y)
                hyper_tuned_q_models[q] = h_model
            return StreamflowModel(hyper_tuned_q_models, pca=pca)

        model = GradientBoostingRegressor(loss='quantile', alpha=.5)
        model.fit(pca_xgb_X, y)
        return StreamflowModel(model, pca=pca)

    else:
        combined_X = pd.concat([xgb_X, xgb_val_X])
        combined_y = pd.concat([y, val_y])
        if quantile:
            hyper_tuned_q_models = {}
            best_q_models = {}
            for q in DEF_QUANTILES:
                h_model = GradientBoostingRegressor(loss='quantile', alpha=q)
                h_model.fit(xgb_X, y)
                hyper_tuned_q_models[q] = h_model
                best_model = GradientBoostingRegressor(loss='quantile', alpha=q)
                best_model.fit(combined_X, combined_y)
                best_q_models[q] = best_model
            return StreamflowModel(hyper_tuned_q_models, pca=None), StreamflowModel(best_q_models, pca=None)

        ht_model = GradientBoostingRegressor(loss='quantile', alpha=.5)
        ht_model.fit(xgb_X, y)
        best_model = GradientBoostingRegressor(loss='quantile', alpha=.5)
        best_model.fit(combined_X, combined_y)
        return StreamflowModel(ht_model, pca=None), StreamflowModel(best_model, pca=None)


def k_nearest_neighbors_fitter(X, y, val_X, val_y, quantile: bool = True):
    assert quantile

    knn_X = base_feature_adapter(X)
    knn_val_X = base_feature_adapter(val_X)
    combined_X = pd.concat([knn_X, knn_val_X])
    combined_y = pd.concat([y, val_y])

    knn = KNeighborsRegressor()

    param_grid = {
        'n_neighbors': np.arange(1, 2500, 250),
    }

    models = {}
    noval_data_models = {}
    for q in DEF_QUANTILES:
        custom_scorer = make_scorer(mean_pinball_loss, greater_is_better=False, alpha=q)

        # Define the grid search
        grid_search = GridSearchCV(knn, param_grid, scoring=custom_scorer)

        # Fit the grid search
        grid_search.fit(combined_X, combined_y)

        models[q] = grid_search.best_estimator_

        noval_model = KNeighborsRegressor(n_neighbors=grid_search.best_params_['n_neighbors'])
        noval_model.fit(knn_X, y)
        noval_data_models[q] = noval_model

    return StreamflowModel(noval_data_models), StreamflowModel(models)


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

        predictor = StreamflowModel(model)
    else:
        regressors = {}
        for q in DEF_QUANTILES:
            qregr = linear_model.QuantileRegressor(quantile=q, alpha=0, solver=solver)
            model = Pipeline([('pca', pca), ('quantile_regression', qregr)])
            model.fit(X, y)

            regressors[q] = model

        predictor = StreamflowModel(regressors)

    return predictor
