import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_pinball_loss
import os

# os.chdir("/Users/emilryd/programming/water-supply-forecast")
os.chdir("..")


def calc_predictive_std(gt: pd.Series, preds: pd.Series) -> float:
    errs = gt - preds
    return errs.std()


def gen_predictive_quantile(preds: pd.Series, std: float, quantile: float) -> pd.Series:
    n_stds = norm.ppf(quantile)
    return preds + n_stds * std


def main(gt_path: str, preds_path: str, gen_std: bool = True):
    ground_truth = pd.read_csv(gt_path)["gt"]
    preds = pd.read_csv(preds_path)["pred"]

    quantiles = [0.1, 0.5, 0.9]
    if gen_std:
        std = calc_predictive_std(ground_truth, preds)
        quantile_preds = {quantile: gen_predictive_quantile(preds, std, quantile) for quantile in quantiles}
    else:
        quantile_preds = {i: preds[f"{i}"].to_numpy() for i in quantiles}

    quantile_losses = {quantile: mean_pinball_loss(ground_truth, q_preds) for quantile, q_preds in
                       quantile_preds.items()}
    interval = (quantile_preds[0.1] <= ground_truth) & (ground_truth <= quantile_preds[0.9])

    print(f'Mean quantile loss:{sum(quantile_losses.values()) / len(quantiles)}, '
          f'percent in 0.1-0.9 quantiles:{100 * np.mean(interval):.2f}%')


if __name__ == '__main__':
    site_ids = pd.read_csv("02-data-cleaning/site_ids.csv")["site_id"]
    for site_id in site_ids:
        main(gt_path=f"model_building_03/model-outputs/ground_truth{site_id}.csv",
             preds_path=f"model_building_03/model-outputs/linear-model-training-optimization/predicted{site_id}.csv")
