import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_pinball_loss
import os
from consts import JULY, DETROIT


from consts import DEF_QUANTILES

os.chdir("../exploration")


def calc_predictive_std(gt: pd.Series, preds: pd.Series) -> float:
    errs = gt - preds
    return errs.std()


def gen_predictive_quantile(preds: pd.Series, std: float, quantile: float) -> pd.Series:
    n_stds = norm.ppf(quantile)
    return preds + n_stds * std


def cache_preds(site_id: str, pred_dates: pd.Series, pred: pd.DataFrame, cache_id: str = None):
    col_order = ['site_id', 'issue_date'] + [f'volume_{int(q * 100)}' for q in DEF_QUANTILES]
    pred_df = pd.DataFrame({"issue_date": pred_dates, 'site_id': site_id} |
                           {f'volume_{int(q * 100)}': pred[q] for q in DEF_QUANTILES})[col_order]

    pred_df.to_csv(f'{cache_id}_pred.csv', index=False)
    return pred_df

def generate_submission_file(ordered_site_ids):
    # Get the correct order, sort in the way competition wants it
    final_submission_df = pd.DataFrame()
    for idx, site_id in enumerate(ordered_site_ids):

        site_submission = pd.read_csv(os.path.join('..', f'{site_id}_pred.csv'))

        if site_id == DETROIT:
            site_submission = site_submission[site_submission.issue_date.dt.month != JULY]
        if idx == 0:
            final_submission_df = pd.concat([final_submission_df, site_submission])

    final_submission_df.site_id = final_submission_df.site_id.astype("category")
    final_submission_df.site_id = final_submission_df.site_id.cat.set_categories(ordered_site_ids)
    final_submission_df = final_submission_df.groupby(final_submission_df.issue_date.dt.year) \
        .apply(lambda x: x.sort_values(['site_id', 'issue_date']))

    final_submission_df.to_csv('final_pred.csv', index=False)
    return final_submission_df
def calc_quantile_loss(gt: pd.Series, preds: pd.DataFrame, quantile: float) -> float:
    return mean_pinball_loss(gt, preds[quantile], alpha=quantile)


def average_quantile_loss(preds: pd.DataFrame, gt: pd.Series, quantiles: list = DEF_QUANTILES) -> float:
    return np.mean([calc_quantile_loss(preds, gt, q) for q in quantiles])


def calc_losses(train_pred: [pd.Series, pd.DataFrame], train_gt: pd.Series, val_pred: [pd.Series, pd.DataFrame],
                val_gt: pd.Series) -> tuple[dict]:
    min_q = min(DEF_QUANTILES)
    max_q = max(DEF_QUANTILES)
    perc_in_interval = {'train': (train_pred[min_q] <= train_gt) & (train_gt <= train_pred[max_q]),
                        'val': (val_pred[min_q] <= val_gt) & (val_gt <= val_pred[max_q])}
    quantile_losses = {'train': {q: calc_quantile_loss(train_gt, train_pred, q) for q in DEF_QUANTILES},
                       'val': {q: calc_quantile_loss(val_gt, val_pred, q) for q in DEF_QUANTILES}}
    avg_q_losses = {'train': average_quantile_loss(train_gt, train_pred, DEF_QUANTILES),
                    'val': average_quantile_loss(val_gt, val_pred, DEF_QUANTILES)}

    return perc_in_interval, quantile_losses, avg_q_losses


def benchmark_results(train_pred: [pd.Series, pd.DataFrame], train_gt: pd.Series, val_pred: [pd.Series, pd.DataFrame],
                      val_gt: pd.Series, test_pred: [pd.Series, pd.DataFrame], benchmark_id: str = None,
                      verbose: bool = False) \
        -> tuple[pd.DataFrame]:
    train_gt = train_gt.reset_index(drop=True)
    val_gt = val_gt.reset_index(drop=True)
    if isinstance(train_pred, pd.DataFrame):
        assert isinstance(val_pred, pd.DataFrame) and isinstance(test_pred, pd.DataFrame), \
            'Error - some of the predictions are for quantiles (as they are dataframes) while others are ' \
            'single predictions!'
        for preds in (train_pred, val_pred, test_pred):
            assert set(preds.columns) == set(DEF_QUANTILES), "Error - pred cols aren't quantiles!"
    else:
        assert isinstance(val_pred, pd.Series) and isinstance(test_pred, pd.Series), \
            'Error - some of the predictions are single predictions (as they are series) while others are ' \
            'for quantiles!'
        # generate predictive std+quantile preds using it, todo see if these preds really are normally distributed
        std = calc_predictive_std(train_gt, train_pred)

        train_pred = pd.DataFrame({q: gen_predictive_quantile(train_pred, std, q) for q in DEF_QUANTILES})
        val_pred = pd.DataFrame({q: gen_predictive_quantile(val_pred, std, q) for q in DEF_QUANTILES})
        test_pred = pd.DataFrame({q: gen_predictive_quantile(test_pred, std, q) for q in DEF_QUANTILES})

    perc_in_interval, quantile_losses, avg_q_losses = calc_losses(train_pred, train_gt, val_pred, val_gt)

    if benchmark_id is not None:
        benchmark_res_path = f'benchmark_res_{benchmark_id}.txt'
        with open(benchmark_res_path, 'w') as f:
            f.write(str(perc_in_interval))
            f.write('\n')
            f.write(str(quantile_losses))
            f.write('\n')
            f.write(str(avg_q_losses))

        if verbose:
            print(f'Percent of preds between {min(DEF_QUANTILES)} and {max(DEF_QUANTILES)} quantiles:'
                  f' {perc_in_interval}\n'
                  f'Quantile losses: {quantile_losses}\n'
                  f'Average quantile loss:{avg_q_losses}')

    return train_pred, val_pred, test_pred
