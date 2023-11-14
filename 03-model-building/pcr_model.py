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

os.chdir("/Users/emilryd/programming/water-supply-forecast")

data = pd.read_csv("02-data-cleaning/training_data.csv")
X = data.values[:,:-3]
print(X)
y = np.reshape(data["volume"].to_numpy(), (-1, 1))

quantiles = [0.1, 0.5, 0.9]


solver = "highs" if sp_version >= parse_version("1.6.0") else "inferior-point"
def quantile_pcr(X, y, pc):
    # Instantiate the PCA object
    pca = PCA()

    #  Preprocessing first derivative
    d1X = savgol_filter(X, 25, polyorder=5, deriv=1)

    # Standardize features removing the mean
    Xstd = StandardScaler().fit_transform(d1X[:, :])

    # Run PCA
    Xreg = pca.fit_transform(Xstd)[:, :pc]
    predictions = {}
    for qu in quantiles:
        qregr = linear_model.QuantileRegressor(quantile=qu, alpha=0.00, solver=solver)
        qregr.fit(Xreg, y)
        y_qc = qregr.predict(Xreg)
        print(f"{qu} -> {np.mean(y_qc>y)}")
        predictions[qu] = y_qc
    
    
    quantile_losses = {quantile: mean_pinball_loss(y, q_preds) for quantile, q_preds in
                       predictions.items()}
    return predictions, sum(quantile_losses.values()) / len(quantiles)


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


quantile = True
# Iterate over every site
site_ids = np.unique(data["site_id"].to_numpy())
for site_id in site_ids:
    mask = np.array((data["site_id"] == site_id))
    mask = np.reshape(mask, (1, -1))
    # print(mask.shape)
    mask = np.transpose(mask)
    masked_X = mask * X    
    masked_y = mask * y

    real_X = np.array([masked_X[idx,:] for idx, i in enumerate(masked_y) if not i==0])
    real_y = np.array([masked_y[idx] for idx, i in enumerate(masked_y) if not i == 0])
    mse_cs = []
    score_cs = []
    mse_cvs = []
    score_cvs = []
    pcs = []
    min_mse = -1
    min_pc = 1
    pred = {}

    for pc in range(1, 30):
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
    
    real_y = np.reshape(real_y, (-1,))
    
    if quantile:
        df = pd.DataFrame({"0.1": pred[0.1], "0.5": pred[0.5], "0.9": pred[0.9], "pcs": min_pc})
        df.to_csv(f"03-model-building/model-outputs/quantile-model-training-optimization/predicted{site_id}.csv", index=False)
    else:
        df = pd.DataFrame({"pred":pred})
        df.to_csv(f"03-model-building/model-outputs/linear-model-training-optimization/predicted{site_id}.csv", index=False)
    
    #gt_df = pd.DataFrame({"gt": real_y})
    #gt_df.to_csv(f"03-model-building/model-outputs/ground_truth{site_id}.csv", index=False)
    
    '''plt.plot(pcs, mse_cvs, 'r')

    plt.xlabel("Principal components")
    plt.ylabel("Log MSE: Training (red), Cross-validation (blue)")
    plt.title(site_id)
    plt.show()'''

    '''plt.plot(score_cs, 'g')  §
    plt.plot(score_cvs, 'b')
    plt.show()'''

