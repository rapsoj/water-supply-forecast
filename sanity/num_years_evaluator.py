import pandas as pd
import os
import matplotlib.pyplot as plt
from pipeline.pipeline import run_pipeline

os.chdir("../exploration")

ground_truth_df = pd.read_csv(os.path.join("..", "assets", "data", "train.csv"))
site_ids = ground_truth_df.site_id.unique()
def get_ave_q_losses():
    sum_train = 0
    sum_val = 0
    for site_id in site_ids:
        avg_q_losses = pd.read_csv(f'{site_id}_avg_q_losses.csv')  # Read perc avg_q_losses file
        sum_train += avg_q_losses.train[0]
        sum_val += avg_q_losses.val[0]

    ave_train = sum_train/len(site_ids)
    ave_val = sum_val/len(site_ids)

    return ave_train, ave_val

if __name__ == '__main__':
    ave_trains = []
    ave_vals = []
    start_years = []
    for start_year in range(1986, 2023):
        start_years.append(start_year)
        run_pipeline(start_year=start_year)
        ave_train, ave_val = get_ave_q_losses()
        ave_trains.append(ave_train)
        ave_vals.append(ave_val)
        print(ave_trains)
        print(ave_vals)

    df = pd.DataFrame(list(zip(ave_trains, ave_vals, start_years)), columns=['train', 'val', 'start_year'])
    df.to_csv('losses_vs_years_xgboost.csv')
    plt.plot(start_years, ave_trains)
    plt.plot(start_years, ave_vals)
    plt.xlabel("Start year")
    plt.ylabel("Average quantile loss (blue: train, orange: val)")
    plt.show()
    plt.pause(1e-3)

