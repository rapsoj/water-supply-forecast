import os
import pandas as pd
import matplotlib.pyplot as plt
from consts import N_PRED_MONTHS, N_PREDS_PER_MONTH, FIRST_FULL_GT_YEAR
from pipeline.pipeline import load_ground_truth

path = os.getcwd()


# Pick your poison: 'test', 'val' or 'train'
data_set = 'test'

num_predictions = N_PRED_MONTHS * N_PREDS_PER_MONTH
validation_years = range(FIRST_FULL_GT_YEAR, 2023, 8)
train_val_gt = load_ground_truth(num_predictions=num_predictions)

val_gt = train_val_gt[train_val_gt.forecast_year.isin(validation_years)]
train_gt = train_val_gt[~train_val_gt.forecast_year.isin(validation_years)]

test_gt = pd.read_csv(os.path.join(path, '..', 'assets', 'data', 'test_monthly_naturalized_flow.csv'))

# Yes I know it's confusing that there is pred and not test here in the csv nae,
# just wanted to keep consistent with previous naming conventions
final_test = pd.read_csv(os.path.join(path, '..', 'exploration', 'final_pred.csv'))
final_val = pd.read_csv(os.path.join(path, '..', 'exploration', 'final_val.csv'))
final_train = pd.read_csv(os.path.join(path, '..', 'exploration', 'final_train.csv'))
site_ids = test_gt.site_id.unique()

# Get training data too

if data_set == 'test':
    gt = test_gt
    final_pred = final_test
elif data_set == 'val':
    gt = val_gt
    final_pred = final_val
elif data_set == 'train':
    gt = train_gt
    final_pred = final_train

final_pred.issue_date = pd.to_datetime(final_pred.issue_date)

for site_id in site_ids:

    site_gt = gt[gt.site_id==site_id].drop(columns=['site_id']).reset_index(drop=True)

    if data_set == 'test': # Approximate test set labels for sanity check
        site_gt = site_gt[site_gt.month.isin((4,5,6))].groupby(['forecast_year']) \
        .agg(lambda x: 1.25*x.sum()).drop(columns=['year', 'month']).reset_index(drop=True)

        site_gt = site_gt.loc[site_gt.index.repeat(num_predictions)]


    site_final_pred = final_pred[final_pred.site_id==site_id].drop(columns=['site_id']).reset_index(drop=True)
    ave_final_pred = site_final_pred.groupby(site_final_pred.issue_date.dt.year).agg(lambda x: x.mean())
    ave_final_pred = ave_final_pred.loc[ave_final_pred.index.repeat(num_predictions)]

    plt.plot(site_final_pred.issue_date, site_final_pred.volume_50, c='r')
    plt.plot(site_final_pred.issue_date, site_gt.volume, 'b')
    plt.plot(ave_final_pred.issue_date, ave_final_pred.volume_50, 'g')
    plt.ylabel("Prediction (full in r, average in g, estimated gt in b)")
    plt.xlabel("Date")
    plt.title(f"{site_id}")
    plt.plot()
    plt.show()