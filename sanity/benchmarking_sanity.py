import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.fit_to_data import Ensemble_Type
from models.fitters import general_pcr_fitter, xgboost_fitter, general_xgboost_fitter, lstm_fitter

os.chdir(os.path.join("..", "outputs"))

fitters = (lstm_fitter,)
models = ('global',)
ensemble_type = Ensemble_Type.BEST_PREDICTION
ground_truth_df = pd.read_csv(os.path.join("..", "assets", "data", "train.csv"))
site_ids = ground_truth_df.site_id.unique()
train_intervs = []
val_intervs = []
ave_train_q_losses = []
ave_val_q_losses = []
train_qlosses = []
val_qlosses = []
ave_gts = []

final_pred = pd.read_csv(os.path.join("..", "outputs", "final_pred.csv"))
#final_pred_local = pd.read_csv(os.path.join("..", "outputs", "final_predlocal.csv"))

pred_sids = final_pred.site_id.unique()

for site_id in site_ids:
    site_gts = []
    site_train_intervs = []
    site_val_intervs = []

    site_train_q_loss = []
    site_val_q_loss = []

    for fitter in fitters:
        for model in models:
            ave_gt = ground_truth_df[ground_truth_df.site_id == site_id].volume.mean()
            site_gts.append(ave_gt)

            perc_interv = pd.read_csv(f'{model}_{fitter.__name__}_{site_id}_perc_in_interval.csv')  # Read perc interval file
            site_train_intervs.append(perc_interv['0'][0])
            site_val_intervs.append(perc_interv['0'][1])

            avg_q_losses = pd.read_csv(f'{model}_{fitter.__name__}_{site_id}_avg_q_losses.csv')  # Read perc avg_q_losses file
            site_train_q_loss.append(avg_q_losses.train[0])
            site_val_q_loss.append(avg_q_losses.val[0])

    if ensemble_type == Ensemble_Type.AVERAGE:
        ave_gts.append(np.mean(site_gts))
        train_intervs.append(np.mean(site_train_intervs))
        val_intervs.append(np.mean(site_val_intervs))
        ave_train_q_losses.append(np.mean(site_train_q_loss))
        ave_val_q_losses.append(np.mean(site_val_q_loss))
    elif ensemble_type == Ensemble_Type.BEST_PREDICTION:
        idx = np.argmin(np.array(site_val_q_loss))
        ave_gts.append(site_gts[idx])
        train_intervs.append(site_train_intervs[idx])
        val_intervs.append(site_val_intervs[idx])
        ave_train_q_losses.append(site_train_q_loss[idx])
        ave_val_q_losses.append(site_val_q_loss[idx])

plt.scatter(site_ids, train_intervs, c='b')
plt.scatter(site_ids, val_intervs, c='r')
print(f'Average interval coverage: {np.mean(val_intervs):.2f}')
plt.ylabel("Percentage in interval")

plt.figure()
plt.scatter(site_ids, ave_train_q_losses)
plt.scatter(site_ids, ave_val_q_losses)
print(f'AMQ validation loss: {np.mean(ave_val_q_losses):.0f}')
print(f'AMQ train loss: {np.mean(ave_train_q_losses):.0f}')

plt.ylabel("Average quantile loss")

plt.figure()
ave_gts = np.array(ave_gts)
ave_train_q_losses = np.array(ave_train_q_losses)
ave_val_q_losses = np.array(ave_val_q_losses)

train_dict = {f'{site_id}': ave_train_q_losses[i] / sum(ave_train_q_losses) for i, site_id in enumerate(site_ids)}
val_dict = {f'{site_id}': ave_val_q_losses[i] / sum(ave_val_q_losses) for i, site_id in enumerate(site_ids)}

df = pd.DataFrame(val_dict, index=np.arange(0, 1)).transpose()
print(df)
plt.scatter(site_ids, ave_train_q_losses / ave_gts)
plt.scatter(site_ids, ave_val_q_losses / ave_gts)
plt.ylabel("Average quantile loss")

plt.show()
