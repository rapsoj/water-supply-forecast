import numpy as np
from scipy.signal import savgol_filter
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.metrics import mean_pinball_loss
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.utils.fixes import parse_version, sp_version

from consts import DEF_QUANTILES


def pcr_fitter(X, y, pc, quantile: bool = True,
               solver="highs" if sp_version >= parse_version("1.6.0") else "inferior-point",
               debug: bool = False):
    # Instantiate the PCA object
    pca = PCA()

    #  Preprocessing first derivative
    d1X = savgol_filter(X, 25, polyorder=5, deriv=1)

    # Standardize features removing the mean
    Xstd = StandardScaler().fit_transform(d1X[:, :])

    # Run PCA
    Xreg = pca.fit_transform(Xstd)[:, :pc]

    if not quantile:
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
    else:
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
