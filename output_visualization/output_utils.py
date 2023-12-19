import pandas as pd
import os
from pipeline.pipeline import load_ground_truth
from consts import N_PRED_MONTHS, N_PREDS_PER_MONTH, FIRST_FULL_GT_YEAR
path = os.getcwd()

def create_val_gt_csv():
    gt = load_ground_truth(N_PRED_MONTHS*N_PREDS_PER_MONTH)
    val_years = range(FIRST_FULL_GT_YEAR, 2023, 8)
    gt_val = gt[gt.forecast_year.isin(val_years)]
    gt_val.to_csv(os.path.join(path, '..', 'outputs', "val_gt.csv"), index=False)

create_val_gt_csv()