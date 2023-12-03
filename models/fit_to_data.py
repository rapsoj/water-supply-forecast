import os

import numpy as np
import pandas as pd

from consts import DEF_QUANTILES
from models.models import pcr_fitter


def fit_basins(fitter: callable, quantile: bool, data: pd.DataFrame, site_ids: list,
               results_path: str = os.path.join('model-outputs',
                                                'model-training-optimization'), max_n_pcs: int = 30):
    results_path = os.path.join(results_path, f'{"quantile" if quantile else "regular"}')
    os.makedirs(results_path, exist_ok=True)

    X = data.drop(columns={'volume', 'site_id', 'forecast_year'})
    y = data.volume
    years = data.forecast_year

    # Iterate over every site
    for site_id in site_ids:
        site_mask = data.site_id == site_id
        masked_X = X[site_mask]
        masked_y = y[site_mask]

        real_X = masked_X[masked_y != 0]  # Get rid of empty labels (from test years, odd 2005-2023)
        real_y = masked_y[masked_y != 0]  # Get rid of empty labels (from test years, odd 2005-2023)

        mse_cvs = []
        pcs = []
        min_mse = -1
        min_pc = 1
        pred = {}

        for pc in range(1, max_n_pcs):
            pcs.append(pc)
            results = fitter(real_X, real_y, pc=pc, quantile=quantile)

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


# todo change pcr fitter internal format
def gen_basin_preds(train_site: pd.DataFrame, train_gt: pd.Series, val_site: pd.DataFrame, val_gt: pd.Series,
                    test: pd.DataFrame, model_fitters=(pcr_fitter,)) -> tuple[pd.Series]:
    if len(model_fitters) != 1:
        raise NotImplementedError('Error - not implemented yet for ensembles!')

    fitter = model_fitters[0]
    model = fitter(train_site, train_gt, val_site, val_gt)
    train_pred = model(train_site)
    val_pred = model(val_site)
    test_pred = model(test)

    return train_pred, val_pred, test_pred


def main():
    os.chdir("../exploration")

    data = pd.read_csv(os.path.join("02-data-cleaning", "transformed_vars.csv"))
    site_id_str = 'site_id_'
    site_ids = [col[col.find(site_id_str) + len(site_id_str):] for col in data.columns]

    fit_basins(fitter=pcr_fitter, quantile=True, data=data, site_ids=site_ids)


if __name__ == '__main__':
    main()
