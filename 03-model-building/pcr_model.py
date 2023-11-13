import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv("training_data.csv")
X = data.values[:,:-3]
y = np.reshape(data["volume"].to_numpy(), (-1, 1))

def pcr(X, y, pc):
    # Instantiate the PCA object
    pca = PCA()

    #  Preprocessing first derivative
    d1X = savgol_filter(X, 25, polyorder=5, deriv=1)

    # Standardize features removing the mea
    Xstd = StandardScaler().fit_transform(d1X[:,:])

    # Run PCA
    Xreg = pca.fit_transform(Xstd)[:,:pc]

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
    score_cv= r2_score(y, y_cv)

    # Mean sqaure error
    mse_c = mean_squared_error(y, y_c)
    mse_cv = mean_squared_error(y, y_cv)

    return(y_cv, score_c, score_cv, mse_c, mse_cv)


# Iterate over every site
site_ids = np.unique(data["site_id"].to_numpy())
for site_id in site_ids:
    mask = np.array((data["site_id"]==site_id))
    mask = np.reshape(mask, (1,-1))
    #print(mask.shape)
    mask = np.transpose(mask)    
    masked_X = mask*X
    masked_y = mask*y
    mse_cs = []
    score_cs = []
    mse_cvs = []
    score_cvs = []
    pcs = []
    for pc in range(1,30):
        pcs.append(pc)
        results = pcr(masked_X,masked_y, pc)
        mse_cs.append(np.log(results[3]))
        mse_cvs.append(np.log(results[4]))
        score_cs.append(results[1])
        score_cvs.append(results[2])
    plt.plot(pcs, mse_cs, 'r')
    plt.plot(pcs, mse_cvs, 'b')
    plt.xlabel("Principal components")
    plt.ylabel("Log MSE: Training (red), Cross-validation (blue)")
    plt.title(site_id)
    plt.show()

    '''plt.plot(score_cs, 'g')
    plt.plot(score_cvs, 'b')
    plt.show()'''



'''# Define the parameter range
parameters = {'pca_n_components': np.arange(1,11,1)}
pcr_pipe = Pipeline([('pca', PCA()), ('linear_regression', LinearRegression())])
pcr = GridSearchCV(pcr_pipe, parameters, scoring = 'neg_mean_squared_error')'''
