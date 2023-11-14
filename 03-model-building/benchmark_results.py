import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_pinball_loss


def calc_predictive_std(gt: pd.Series, preds: pd.Series) -> float:
    errs = gt - preds
    return errs.std()


def gen_predictive_quantile(preds: pd.Series, std: float, quantile: float) -> pd.Series:
    n_stds = norm.ppf(quantile)
    return preds + n_stds * std


def main(gt_path: str, preds_path: str):
    ground_truth = pd.read_csv(gt_path)
    preds = pd.read_csv(preds_path)

    std = calc_predictive_std(ground_truth, preds)
    quantiles = [0.1, 0.5, 0.9]
    quantile_preds = {quantile: gen_predictive_quantile(preds, std, quantile) for quantile in quantiles}
    quantile_losses = {quantile: mean_pinball_loss(ground_truth, q_preds) for quantile, q_preds in
                       quantile_preds.items()}
    print(f'Mean quantile loss:{sum(quantile_losses.values()) / len(quantiles)}, '
          f'percent in 0.1-0.9 quantiles:{100 * np.mean(quantile_preds[0.1] <= preds <= quantile_preds[0.9]):.2f}%')


if __name__ == '__main__':
    main()
