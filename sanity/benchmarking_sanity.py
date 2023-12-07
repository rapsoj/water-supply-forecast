import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

os.chdir("../exploration")

ground_truth_df = pd.read_csv(os.path.join("..", "assets", "data", "train.csv"))
site_ids = ground_truth_df.site_id.unique()
train_intervs = []
val_intervs = []
ave_train_q_losses = []
ave_val_q_losses = []
train_qlosses = []
val_qlosses = []
for site_id in site_ids:
    perc_interv = pd.read_csv(f'{site_id}_perc_in_interval.csv')  # Read perc interval file
    train_intervs.append(perc_interv['0'][0])
    val_intervs.append(perc_interv['0'][1])
    avg_q_losses = pd.read_csv(f'{site_id}_avg_q_losses.csv')  # Read perc avg_q_losses file
    ave_train_q_losses.append(avg_q_losses.train[0])
    ave_val_q_losses.append(avg_q_losses.val[0])

    all_q_losses = pd.read_csv(f'{site_id}_all_q_losses.csv')  # Read perc avg_q_losses file
    train_qlosses.append(all_q_losses.train.to_numpy())
    val_qlosses.append(all_q_losses.val)

plt.scatter(site_ids, train_intervs, c='b')
plt.scatter(site_ids, val_intervs, c='r')
print(np.mean(val_intervs))
plt.ylabel("Percentage in interval")

plt.figure()
plt.scatter(site_ids, ave_train_q_losses)
plt.scatter(site_ids, ave_val_q_losses)
print(np.mean(ave_val_q_losses))
print(np.mean(ave_train_q_losses))

plt.ylabel("Average quantile loss")

plt.figure()
plt.scatter(site_ids, np.array(train_qlosses)[:, 0], c='r')
plt.scatter(site_ids, np.array(train_qlosses)[:, 1], c='b')
plt.scatter(site_ids, np.array(train_qlosses)[:, 2], c='g')
plt.ylabel("Quantile loss per site r:0.1, b:0.5, g:0.9")

plt.show()
