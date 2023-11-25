import pandas as pd
import numpy as np

# Extracting data from metatadata and train csv files
data = pd.read_csv("Data/metadata.csv")
site_ids = data["usgs_id"]
for idx, i in enumerate(site_ids):
    if not np.isnan(i):
        site_ids[idx] = int(i)
df = pd.DataFrame({"site_id": site_ids, "site_name":data["usgs_name"]})
df.to_csv("Data/site_usgs_ids.csv", index=False)
print(site_ids)


'''# Extracting from Excel files of site data downloaded from USGS NWIS
usgs_data = pd.read_excel('/Users/emilryd/Downloads/USGS_site_data_batch3.xlsx', sheet_name=None)

# Loop over sheets
for sheet_name, df in usgs_data.items():
    df.to_csv(f"{sheet_name}.csv")'''