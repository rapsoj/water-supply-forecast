import os
import pandas as pd

path = os.getcwd()

monthly_flow = pd.read_csv(os.path.join(path, '..', 'data', 'additional_sites', 'train_monthly_naturalized_flow'))
monthly_flow['site_id'] = monthly

gt = monthly_flow.groupby()