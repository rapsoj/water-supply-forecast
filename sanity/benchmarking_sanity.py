import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from models.models import general_pcr_fitter, xgboost_fitter

os.chdir(os.path.join("..", "pipeline"))

fitters = (general_pcr_fitter, xgboost_fitter)
models = ('local', 'global')

ground_truth_df = pd.read_csv(os.path.join("..", "assets", "data", "train.csv"))
site_ids = ground_truth_df.site_id.unique()
train_intervs = []
val_intervs = []
ave_train_q_losses = []
ave_val_q_losses = []
train_qlosses = []
val_qlosses = []
ave_gts = []
for site_id in site_ids:
    site_sum_gts = 0
    site_sum_train_intervs = 0
    site_sum_val_intervs = 0

    site_sum_train_q_loss = 0
    site_sum_val_q_loss = 0

    for fitter in fitters:
        for model in models:
            if not (model == 'global' and fitter.__name__ == 'general_pcr_fitter'):
                ave_gt = ground_truth_df[ground_truth_df.site_id == site_id].volume.mean()
                site_sum_gts += ave_gt

                perc_interv = pd.read_csv(f'{model}_{fitter.__name__}_{site_id}_perc_in_interval.csv')
                site_sum_train_intervs += perc_interv['0'][0]
                site_sum_val_intervs += perc_interv['0'][1]

                avg_q_losses = pd.read_csv(
                    f'{model}_{fitter.__name__}_{site_id}_avg_q_losses.csv')  # Read perc avg_q_losses file
                site_sum_train_q_loss += avg_q_losses.train[0]
                site_sum_val_q_loss += avg_q_losses.val[0]

                '''all_q_losses = pd.read_csv(f'{model}_{site_id}_{fitter}_all_q_losses.csv')  # Read perc avg_q_losses file
                train_qlosses.append(all_q_losses.train.to_numpy())
                val_qlosses.append(all_q_losses.val)'''
    ave_gts.append(site_sum_gts / (len(fitters) * len(models)))
    train_intervs.append(site_sum_train_intervs / (len(fitters) * len(models)))
    val_intervs.append(site_sum_val_intervs / (len(fitters) * len(models)))
    ave_train_q_losses.append(site_sum_train_q_loss / (len(fitters) * len(models)))
    ave_val_q_losses.append(site_sum_val_q_loss / (len(fitters) * len(models)))

plt.scatter(site_ids, train_intervs, c='b')
plt.scatter(site_ids, val_intervs, c='r')
print(np.mean(val_intervs))
plt.ylabel("Percentage in interval")

plt.figure()
plt.scatter(site_ids, ave_train_q_losses)
plt.scatter(site_ids, ave_val_q_losses)
print(np.mean(ave_val_q_losses))
print(np.mean(ave_train_q_losses))
print(site_ids[np.argmax(ave_train_q_losses)])

plt.ylabel("Average quantile loss")

plt.figure()
ave_gts = np.array(ave_gts)
ave_train_q_losses = np.array(ave_train_q_losses)
ave_val_q_losses = np.array(ave_val_q_losses)

train_dict = {f'{site_id}': ave_train_q_losses[i] / sum(ave_train_q_losses) for i, site_id in enumerate(site_ids)}
val_dict = {f'{site_id}': ave_val_q_losses[i] / sum(ave_val_q_losses) for i, site_id in enumerate(site_ids)}

df = pd.DataFrame(val_dict, index=np.arange(0, 1)).transpose()
print(df)
# print(my_dict)
plt.scatter(site_ids, ave_train_q_losses / ave_gts)
plt.scatter(site_ids, ave_val_q_losses / ave_gts)
plt.ylabel("Average quantile loss")

'''plt.figure()
plt.scatter(site_ids, np.array(train_qlosses)[:, 0], c='r')
plt.scatter(site_ids, np.array(train_qlosses)[:, 1], c='b')
plt.scatter(site_ids, np.array(train_qlosses)[:, 2], c='g')
plt.ylabel("Quantile loss per site r:0.1, b:0.5, g:0.9")'''

plt.show()
