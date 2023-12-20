import os

import numpy as np
import pandas as pd
from consts import JULY, APRIL, FIRST_FULL_GT_YEAR, NUM_MONTHS_IN_SEASON

path = os.getcwd()

monthly_flow = pd.read_csv(
    os.path.join(path, '..', 'assets', 'data', 'additional_sites', 'train_monthly_naturalized_flow.csv'))
metadata = pd.read_csv(os.path.join(path, '..', 'assets', 'data', 'additional_sites', 'metadata.csv'))

monthly_flow = monthly_flow[(monthly_flow.month <= JULY) & (monthly_flow.month >= APRIL)]

outlier_sites = [np.nan]  # todo manually find high/low streamflow sites

monthly_flow = monthly_flow[~monthly_flow.site_id.isin(outlier_sites)]

min_perc_existing_months = 0.5
nan_site_years = monthly_flow.groupby(['site_id', 'year']) \
    .volume \
    .apply(lambda x: x.name if x.isna().mean() > min_perc_existing_months else np.nan) \
    .dropna() \
    .values
monthly_flow = monthly_flow[~monthly_flow.set_index(['site_id', 'year']).index.isin(nan_site_years)] \
    .reset_index(drop=True)


def fill_sitewise_monthly_mean(row: pd.Series, sitewise_month_means: dict) -> pd.DataFrame:
    return sitewise_month_means[(row.site_id, row.month)]


sitewise_month_means = monthly_flow[~monthly_flow.volume.isna()].groupby(['site_id', 'month']).volume.mean().to_dict()
monthly_flow.volume[monthly_flow.volume.isna()] = monthly_flow[monthly_flow.volume.isna()] \
    .apply(fill_sitewise_monthly_mean, sitewise_month_means=sitewise_month_means, axis='columns')

gt = monthly_flow.groupby(['site_id', 'year'], as_index=False).volume.sum()

site_ids = gt.site_id.unique()

gt = gt[gt.year >= FIRST_FULL_GT_YEAR]
gt = gt[gt.volume != 0]

gt.site_id = gt.site_id.str.lower() \
    .replace('[ ]', '_', regex=True)

gt.to_csv(os.path.join(path, '..', 'assets', 'data', 'additional_train.csv'))
