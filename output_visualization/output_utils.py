import pandas as pd
import os
from pipeline.pipeline import load_ground_truth
from consts import N_PRED_MONTHS, N_PREDS_PER_MONTH, FIRST_FULL_GT_YEAR, ORDERED_SITE_IDS, DETROIT
path = os.getcwd()

def create_val_gt_csv():
    gt = load_ground_truth(N_PRED_MONTHS*N_PREDS_PER_MONTH)
    val_years = range(FIRST_FULL_GT_YEAR, 2023, 8)
    gt_val = gt[gt.forecast_year.isin(val_years)]
    # Remove last four detroit sites for every year
    gt_detroit = gt_val[gt_val.site_id==DETROIT].drop_duplicates()
    gt_detroit = gt_detroit.loc[gt_detroit.index.repeat((N_PRED_MONTHS-1)*(N_PREDS_PER_MONTH))]
    gt_val = gt_val[~(gt_val.site_id==DETROIT)].reset_index(drop=True)
    gt_val = pd.concat([gt_detroit, gt_val])


    gt_val.site_id = gt_val.site_id.astype("category")
    gt_val.site_id = gt_val.site_id.cat.set_categories(ORDERED_SITE_IDS)
    gt_val = gt_val.groupby(gt_val.forecast_year) \
        .apply(lambda x: x.sort_values(['site_id']))



    gt_val.to_csv(os.path.join(path, '..', 'models', 'calibration_data', "val_gt.csv"), index=False)

create_val_gt_csv()