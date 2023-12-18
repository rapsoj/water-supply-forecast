import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_pinball_loss, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from benchmark.benchmark_results import average_quantile_loss
from consts import DEF_QUANTILES, JULY
from models.lstm_utils import features2seqs, pad_collate_fn, train_lstm, DEF_LSTM_HYPPARAMS


def base_feature_adapter(X, pca=None):
    X = (X[X.date.dt.month <= JULY].drop(columns=['date', 'forecast_year']))
    return X


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=DEF_LSTM_HYPPARAMS.hidden_size, num_layers=DEF_LSTM_HYPPARAMS.n_hidden,
                 output_size=2, dropout_prob: float = DEF_LSTM_HYPPARAMS.dropout_prob):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # Pack the padded sequences
        x_packed = pack_padded_sequence(x, lengths, batch_first=True)
        out_packed, _ = self.lstm(x_packed)
        out_padded, _ = pad_packed_sequence(out_packed, batch_first=True)
        # Apply the linear layer to the unpacked outputs
        out = self.fc(out_padded)
        outputs = out[torch.arange(out.size(0)), np.array(lengths) - 1]  # Return the outputs for the last time step
        means, log_vars = outputs[:, 0], outputs[:, 1]
        return means, torch.exp(0.5 * log_vars)


class StreamflowModel:
    def __init__(self, model, adapter=base_feature_adapter, pca=None):
        self.model = model
        self.adapter = adapter
        self._loss = average_quantile_loss if isinstance(self.model, dict) else mean_squared_error
        self.pca = pca

    def __call__(self, X: pd.DataFrame, *args, adapt_feats: bool = True):
        if adapt_feats:
            X = self.adapter(X)

        if isinstance(X, pd.DataFrame):
            assert (X.dtypes == float).all(), 'Error - wrong dtypes!'
        else:
            pass  # todo add some asserts for the lstm case?

        if self.pca is not None:  # for non PCR models
            X = self.pca.transform(X)

        if isinstance(self.model, dict):
            pred = pd.DataFrame({q: q_model.predict(X) for q, q_model in self.model.items()})
        elif isinstance(self.model, nn.Module):
            pred = []
            for sequences, lengths in X:
                means, stds = self.model(sequences, lengths)
                # todo create function for this
                pred.append(pd.DataFrame({q: (means + norm.ppf(q) * stds).item() for q in DEF_QUANTILES}, index=[0]))
            pred = pd.concat(pred).reset_index(drop=True)
        else:
            pred = pd.Series(self.model.predict(X))

        return pred

    def loss(self, X, y, adapt_feats: bool = True):
        pred = self(X, adapt_feats=adapt_feats)
        assert pred.shape[0] == y.shape[0], 'Error - predictions/ground truth mismatch!'

        loss = self._loss(y, pred)
        assert loss is not None and loss is not np.nan, 'Error - loss is None!'
        return loss


def lstm_fitter(X, y, val_X, val_y, quantile: bool = True):
    assert quantile

    train_set = features2seqs(X, y)
    val_set = features2seqs(val_X, val_y)
    combined_set = features2seqs(pd.concat([X, val_X]).reset_index(drop=True),
                                 pd.concat([y, val_y]).reset_index(drop=True))

    n_feats = train_set[0][0].shape[1]

    train_dloader = DataLoader(train_set, batch_size=DEF_LSTM_HYPPARAMS.bs, shuffle=True, collate_fn=pad_collate_fn)
    full_dloader = DataLoader(combined_set, batch_size=DEF_LSTM_HYPPARAMS.bs, shuffle=True, collate_fn=pad_collate_fn)

    train_model = LSTMModel(input_size=n_feats)
    full_model = LSTMModel(input_size=n_feats)

    full_model = train_lstm(full_dloader, None, full_model, DEF_LSTM_HYPPARAMS.lr, DEF_LSTM_HYPPARAMS.n_epochs)


    train_model = train_lstm(train_dloader, val_set, train_model, DEF_LSTM_HYPPARAMS.lr, DEF_LSTM_HYPPARAMS.n_epochs)
    # todo try fine-tuning trained model? probably no reason to do that, difficult to validate

    def lstm_feat_adapter(X):
        dataset = features2seqs(X)
        dataloader = DataLoader(dataset, collate_fn=pad_collate_fn)

        return dataloader

    train_model.eval()
    full_model.eval()
    return StreamflowModel(train_model, adapter=lstm_feat_adapter), \
           StreamflowModel(full_model, adapter=lstm_feat_adapter)


def general_xgboost_fitter(X, y, val_X, val_y, MAX_N_PCS=30, quantile: bool = True, using_pca=True):
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
