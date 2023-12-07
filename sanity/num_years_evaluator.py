import pandas as pd
from pipeline.pipeline import run_pipeline

if __name__ == '__main__':
    for start_year in range(1986,2023):
        run_pipeline(start_year=start_year)


def get_ave_q_losses():
    for site_id in site_ids:
        all_q_losses = pd.read_csv(f'{site_id}_all_q_losses.csv')  # Read perc avg_q_losses file
