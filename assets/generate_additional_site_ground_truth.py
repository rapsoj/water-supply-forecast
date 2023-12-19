import os
import pandas as pd
from consts import JULY, APRIL, FIRST_FULL_GT_YEAR, NUM_MONTHS_IN_SEASON
path = os.getcwd()

monthly_flow = pd.read_csv(os.path.join(path, '..', 'assets', 'data', 'additional_sites', 'train_monthly_naturalized_flow.csv'))
metadata = pd.read_csv(os.path.join(path, '..', 'assets', 'data', 'additional_sites', 'metadata.csv'))

monthly_flow = monthly_flow[(monthly_flow.month <= JULY) & (monthly_flow.month >= APRIL)]

def process_zeros(df: pd.DataFrame) -> pd.DataFrame:
    # todo fill nans with sitewise monthwise average
    return df

monthly_flow = monthly_flow.apply(process_zeros)
monthly_flow.volume = monthly_flow.volume.fillna(0)

gt = monthly_flow.groupby(['nrcs_id', 'year'], as_index=False).volume.sum()

site_ids = [metadata.nrcs_name[metadata.nrcs_id==x].iloc[0] for x in gt.nrcs_id]

gt['site_id'] = pd.Series(data=site_ids)
gt = gt[gt.year >= FIRST_FULL_GT_YEAR]
gt = gt[gt.volume != 0]

gt.site_id = gt.site_id.str.lower() \
    .replace('[ ]', '_', regex=True)

gt = gt.drop(columns='nrcs_id')
gt.to_csv(os.path.join(path, '..', 'assets', 'data', 'additional_train.csv'))

