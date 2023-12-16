import os
import pandas as pd
import matplotlib.pyplot as plt
from consts import N_PRED_MONTHS, N_PREDS_PER_MONTH
path = os.getcwd()


gt = pd.read_csv(os.path.join(path, '..', 'assets', 'data', 'test_monthly_naturalized_flow.csv'))
final_pred = pd.read_csv(os.path.join(path, '..', 'exploration', 'final_pred.csv'))
final_pred.issue_date = pd.to_datetime(final_pred.issue_date)
site_ids = gt.site_id.unique()




for site_id in site_ids:


    site_gt = gt[gt.site_id==site_id].drop(columns=['site_id']).reset_index(drop=True)


    site_gt = site_gt[site_gt.month.isin((4,5,6))].groupby(['forecast_year']) \
        .agg(lambda x: 1.25*x.sum()).drop(columns=['year', 'month']).reset_index(drop=True)
    num_predictions = N_PRED_MONTHS * N_PREDS_PER_MONTH
    site_gt = site_gt.loc[site_gt.index.repeat(num_predictions)]

    site_final_pred = final_pred[final_pred.site_id==site_id].drop(columns=['site_id']).reset_index(drop=True)

    ave_final_pred = site_final_pred.groupby(site_final_pred.issue_date.dt.year).agg(lambda x: x.mean())
    ave_final_pred = ave_final_pred.loc[ave_final_pred.index.repeat(num_predictions)]
    plt.scatter(site_final_pred.issue_date, site_final_pred.volume_50, c='r')
    plt.plot(site_final_pred.issue_date, site_gt.volume, 'b')
    plt.plot(ave_final_pred.issue_date, ave_final_pred.volume_50, 'g')
    plt.ylabel("Prediction (full in r, average in g, estimated gt in b")
    plt.xlabel("Date")
    plt.title(f"{site_id}")
    plt.plot()
    plt.show()