import numpy as np
import pandas as pd

from consts import DEF_QUANTILES
from models.models import xgboost_fitter, k_nearest_neighbors_fitter, general_pcr_fitter


def data_gen(n_samples: int, out_dim: int = 15) -> tuple:
    x = np.random.randn(n_samples)
    X = np.zeros((n_samples, out_dim))
    X[:, 0] = x
    y = x

    X = pd.DataFrame(X)
    X['date'] = pd.to_datetime('2020-01-01')
    X['forecast_year'] = np.nan

    return X, pd.Series(y)


def straight_line_err_is_good(fitter=general_pcr_fitter, train_size: int = 5000, val_size: int = 1000,
                              test_size: int = 1000):
    X, y = data_gen(train_size)
    val_X, val_y = data_gen(val_size)

    train_model, full_model = fitter(X, y, val_X, val_y)

    test_X, test_y = data_gen(test_size)
    test_pred = full_model(test_X)

    print('RMSE for different quantiles:')
    for q in DEF_QUANTILES:
        print(f'{q}:{((test_pred[q] - test_y) ** 2).mean()}')


if __name__ == '__main__':
    straight_line_err_is_good()
