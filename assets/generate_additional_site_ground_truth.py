import os

import numpy as np
import pandas as pd
from consts import JULY, APRIL, FIRST_FULL_GT_YEAR, NUM_MONTHS_IN_SEASON

path = os.getcwd()

monthly_flow = pd.read_csv(
    os.path.join(path, '..', 'assets', 'data', 'additional_sites', 'train_monthly_naturalized_flow.csv'))
metadata = pd.read_csv(os.path.join(path, '..', 'assets', 'data', 'additional_sites', 'metadata.csv'))

monthly_flow = monthly_flow[(monthly_flow.month <= JULY) & (monthly_flow.month >= APRIL)]

nan_sites = monthly_flow.groupby('nrcs_id') \
    .volume \
    .apply(lambda x: x.name if x.isna().all() else np.nan) \
    .dropna() \
    .values
outlier_sites = []  # todo manually find high/low streamflow sites

monthly_flow = monthly_flow[(~monthly_flow.nrcs_id.isin(nan_sites)) |
                            (~monthly_flow.nrcs_id.isin(outlier_sites))]


def fill_sitewise_monthly_mean(row: pd.Series, sitewise_month_means: dict) -> pd.DataFrame:
    return sitewise_month_means[(row.nrcs_id, row.month)]


sitewise_month_means = monthly_flow[~monthly_flow.volume.isna()].groupby(['nrcs_id', 'month']).volume.mean().to_dict()
monthly_flow.volume[monthly_flow.volume.isna()] = monthly_flow[monthly_flow.volume.isna()] \
    .apply(fill_sitewise_monthly_mean, sitewise_month_means=sitewise_month_means, axis='columns')

gt = monthly_flow.groupby(['nrcs_id', 'year'], as_index=False).volume.sum()

site_ids = [metadata.nrcs_name[metadata.nrcs_id == x].iloc[0] for x in gt.nrcs_id]

gt['site_id'] = pd.Series(data=site_ids)
gt = gt[gt.year >= FIRST_FULL_GT_YEAR]
gt = gt[gt.volume != 0]

gt.site_id = gt.site_id.str.lower() \
    .replace('[ ]', '_', regex=True)

gt = gt.drop(columns='nrcs_id')
gt.to_csv(os.path.join(path, '..', 'assets', 'data', 'additional_train.csv'))
