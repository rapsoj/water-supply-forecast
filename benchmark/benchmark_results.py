import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_pinball_loss
import os
from consts import DEF_QUANTILES, JULY, DETROIT, CODESPACE_RUN

if CODESPACE_RUN:
    os.chdir('outputs')
else:
    os.chdir(os.path.join('..', 'outputs'))


# todo be smarter than using an error threshold
def calc_predictive_std(gt: pd.Series, preds: pd.Series) -> float:
    errs = gt - preds
    return errs.std()


def gen_predictive_quantile(preds: pd.Series, std: float, quantile: float) -> pd.Series:
    n_stds = norm.ppf(quantile)
    return preds + n_stds * std


def cache_preds(site_id: str, pred_dates: pd.Series, pred: pd.DataFrame, set_id: str, cache_id: str = None):
    col_order = ['site_id', 'issue_date'] + [f'volume_{int(q * 100)}' for q in DEF_QUANTILES]
    pred_df = pd.DataFrame({"issue_date": pred_dates, 'site_id': site_id} |
                           {f'volume_{int(q * 100)}': pred[q] for q in DEF_QUANTILES})[col_order]

    pred_df.to_csv(f'{cache_id}_{set_id}.csv', index=False)
    return pred_df


def generate_submission_file(ordered_site_ids, model_id: str, fitter_id: str, set_id: str):
    # Get the correct order, sort in the way competition wants it
    final_submission_df = pd.DataFrame()
    for idx, site_id in enumerate(ordered_site_ids):
        site_submission = pd.read_csv(f'{model_id}_{fitter_id}_{site_id}_{set_id}.csv')
        # todo explicitly pass this as an argument
        # todo get this path from a func instead of hardcoding+copy pasting
        site_submission.issue_date = site_submission.issue_date.astype('datetime64[ns]')
        if site_id == DETROIT:
            site_submission = site_submission[site_submission.issue_date.dt.month != JULY]
        final_submission_df = pd.concat([final_submission_df, site_submission])

    final_submission_df.site_id = final_submission_df.site_id.astype("category")
    final_submission_df.site_id = final_submission_df.site_id.cat.set_categories(ordered_site_ids)
    final_submission_df = final_submission_df.groupby(final_submission_df.issue_date.dt.year) \
        .apply(lambda x: x.sort_values(['site_id', 'issue_date']))

    final_submission_df.to_csv(f'final_pred_{model_id}_{fitter_id}.csv', index=False)
    return final_submission_df


def cache_merged_submission_file(df: pd.DataFrame, label: str):
    df.to_csv(f'final_{label}.csv', index=False)


def calc_quantile_loss(gt: pd.Series, preds: pd.DataFrame, quantile: float) -> float:
    return 2 * mean_pinball_loss(gt, preds[quantile], alpha=quantile)  # Factor 2 to match DrivenData


def average_quantile_loss(gt: pd.Series, preds: pd.DataFrame, quantiles: list = DEF_QUANTILES) -> float:
    return np.mean([calc_quantile_loss(gt, preds, q) for q in quantiles])


def calc_losses(train_pred: [pd.Series, pd.DataFrame], train_gt: pd.Series, val_pred: [pd.Series, pd.DataFrame],
                val_gt: pd.Series) -> tuple[dict]:
    min_q = min(DEF_QUANTILES)
    max_q = max(DEF_QUANTILES)
    if train_pred.empty:
        perc_in_interval = {'train': -1,
                            'val': (val_pred[min_q] <= val_gt) & (val_gt <= val_pred[max_q])}
        quantile_losses = {'train': {q: -1 for q in DEF_QUANTILES},
                           'val': {q: calc_quantile_loss(val_gt, val_pred, q) for q in DEF_QUANTILES}}
        avg_q_losses = {'train': -1,
                        'val': average_quantile_loss(val_gt, val_pred, DEF_QUANTILES)}
    elif val_pred.empty:
        perc_in_interval = {'train': (train_pred[min_q] <= train_gt) & (train_gt <= train_pred[max_q]),
                            'val': -1}
        quantile_losses = {'train': {q: calc_quantile_loss(train_gt, train_pred, q) for q in DEF_QUANTILES},
                           'val': {q: -1 for q in DEF_QUANTILES}}
        avg_q_losses = {'train': average_quantile_loss(train_gt, train_pred, DEF_QUANTILES),
                        'val': -1}
    else:
        perc_in_interval = {'train': (train_pred[min_q] <= train_gt) & (train_gt <= train_pred[max_q]),
                            'val': (val_pred[min_q] <= val_gt) & (val_gt <= val_pred[max_q])}
        quantile_losses = {'train': {q: calc_quantile_loss(train_gt, train_pred, q) for q in DEF_QUANTILES},
                           'val': {q: calc_quantile_loss(val_gt, val_pred, q) for q in DEF_QUANTILES}}
        avg_q_losses = {'train': average_quantile_loss(train_gt, train_pred, DEF_QUANTILES),
                        'val': average_quantile_loss(val_gt, val_pred, DEF_QUANTILES)}

    return perc_in_interval, quantile_losses, avg_q_losses


def quantilise_preds(train_pred, val_pred, test_pred, train_gt):
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

    return train_pred, val_pred, test_pred


def benchmark_results(train_pred: [pd.Series, pd.DataFrame], train_gt: pd.Series, val_pred: [pd.Series, pd.DataFrame],
                      val_gt: pd.Series, benchmark_id: str = None, verbose: bool = False) \
        -> tuple[pd.DataFrame]:
    # todo make sure we can pass this assertion of non-negative predictions
    # assert (train_pred >= 0).all().all() and (val_pred >= 0).all().all() and (test_pred >= 0).all().all(), \
    #     'Error - negative predictions!'
    assert (train_gt >= 0).all() and (val_gt >= 0).all(), 'Error - negative ground truths!'

    # for data in (train_pred, val_pred, test_pred):
    #    assert all((data[DEF_QUANTILES[i]] <= data[DEF_QUANTILES[i + 1]]).all()
    #               for i in range(len(DEF_QUANTILES) - 1)), 'Error - quantiles are not ordered!'

    perc_in_interval, quantile_losses, avg_q_losses = calc_losses(train_pred, train_gt, val_pred, val_gt)

    # Extract and write to csvs to do sanity checks
    avg_q_losses = pd.DataFrame(avg_q_losses, index=np.arange(0, 1))
    quantile_losses = pd.DataFrame(quantile_losses)

    perc_in_interval = pd.DataFrame(perc_in_interval)
    perc_in_interval = perc_in_interval.mean(axis='rows')
    avg_q_losses.to_csv(f'{benchmark_id}_avg_q_losses.csv', index=False)
    quantile_losses.to_csv(f'{benchmark_id}_all_q_losses.csv', index=False)
    perc_in_interval.to_csv(f'{benchmark_id}_perc_in_interval.csv', index=False)

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
