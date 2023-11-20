import os

import numpy as np
import pandas as pd

from model_building_03.models import pcr_fitter
from consts import DEF_QUANTILES


def fit_basins(fitter: callable, quantile: bool, data: pd.DataFrame,
               results_path: str = os.path.join('model_building_03', 'model-outputs',
                                                'model-training-optimization'), max_n_pcs: int = 30):
    results_path = os.path.join(results_path, f'{"quantile" if quantile else "regular"}')
    os.makedirs(results_path, exist_ok=True)

    X = data.drop(columns={'volume', 'site_id', 'forecast_year'})
    y = data.volume

    # Iterate over every site
    site_ids = data.site_id.unique()
    for site_id in site_ids:
        site_mask = data.site_id == site_id
        masked_X = X[site_mask]
        masked_y = y[site_mask]

        real_X = masked_X[masked_y != 0]
        real_y = masked_y[masked_y != 0]
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


def main():
    os.chdir("..")

    data = pd.read_csv(os.path.join("02-data-cleaning", "training_data.csv"))

    fit_basins(fitter=pcr_fitter, quantile=True, data=data)


if __name__ == '__main__':
    main()
