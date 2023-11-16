import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils.fixes import parse_version, sp_version
from sklearn.metrics import mean_pinball_loss
import os

from consts import DEF_QUANTILES


def quantile_pcr(X, y, pc, solver="highs" if sp_version >= parse_version("1.6.0") else "inferior-point",
                 debug: bool = False):
    # Instantiate the PCA object
    pca = PCA()

    #  Preprocessing first derivative
    d1X = savgol_filter(X, 25, polyorder=5, deriv=1)

    # Standardize features removing the mean
    Xstd = StandardScaler().fit_transform(d1X[:, :])

    # Run PCA
    Xreg = pca.fit_transform(Xstd)[:, :pc]
    predictions = {}
    for qu in DEF_QUANTILES:
        qregr = linear_model.QuantileRegressor(quantile=qu, alpha=0.00, solver=solver)
        qregr.fit(Xreg, y)
        y_qc = qregr.predict(Xreg)

        if debug:
            print(f"{qu} -> {np.mean(y_qc > y)}")

        predictions[qu] = y_qc

    quantile_losses = {quantile: mean_pinball_loss(y, q_preds) for quantile, q_preds in
                       predictions.items()}
    return predictions, sum(quantile_losses.values()) / len(DEF_QUANTILES)


def mean_pcr(X, y, pc):
    # Instantiate the PCA object
    pca = PCA()

    #  Preprocessing first derivative
    d1X = savgol_filter(X, 25, polyorder=5, deriv=1)

    # Standardize features removing the mean
    Xstd = StandardScaler().fit_transform(d1X[:, :])

    # Run PCA
    Xreg = pca.fit_transform(Xstd)[:, :pc]

    # Instantiate linear regression object
    regr = linear_model.LinearRegression()

    # Fit
    regr.fit(Xreg, y)

    # Calibrate
    y_c = regr.predict(Xreg)

    # Cross-validation
    y_cv = cross_val_predict(regr, Xreg, y, cv=10)

    # Scores
    score_c = r2_score(y, y_c)
    score_cv = r2_score(y, y_cv)

    # Mean square error
    mse_c = mean_squared_error(y, y_c)
    mse_cv = mean_squared_error(y, y_cv)

    return y_c, y_cv, score_c, score_cv, mse_c, mse_cv


def fit_basins(quantile: bool, data: pd.DataFrame,
               results_path: str = os.path.join('03-model-building', 'model-outputs',
                                                'model-training-optimization-new'), max_n_pcs: int = 30):
    results_path = os.path.join(results_path, f'{"quantile" if quantile else "regular"}')
    os.makedirs(results_path, exist_ok=True)

    X = data.drop(columns={'volume', 'site_id', 'forecast_year'})
    y = data.volume

    # Iterate over every site
    site_ids = data.site_id.unique()
    for site_id in site_ids:
        mask = data.site_id == site_id
        masked_X = X[mask]
        masked_y = y[mask]

        real_X = masked_X[masked_y != 0]
        real_y = masked_y[masked_y != 0]
        mse_cvs = []
        pcs = []
        min_mse = -1
        min_pc = 1
        pred = {}

        for pc in range(1, max_n_pcs):
            pcs.append(pc)
            if quantile:
                results = quantile_pcr(real_X, real_y, pc)
            else:
                results = mean_pcr(real_X, real_y, pc)

            if results[-1] < min_mse or pc == 1:
                min_mse = results[-1]
                pred = results[0]
                min_pc = pc
            mse_cvs.append(results[-1])

        if not quantile:
            pred = np.reshape(pred, (-1,))

        site_path = os.path.join(results_path, f'predicted_{site_id}.csv')
        if quantile:
            res = {q: pred[q] for q in DEF_QUANTILES}
            res["pcs"] = min_pc
            df = pd.DataFrame(res)
        else:
            df = pd.DataFrame({"pred": pred})
        df.to_csv(site_path, index=False)


def main():
    os.chdir("..")

    data = pd.read_csv(os.path.join("02-data-cleaning", "training_data.csv"))

    fit_basins(quantile=True, data=data)


if __name__ == '__main__':
    main()
