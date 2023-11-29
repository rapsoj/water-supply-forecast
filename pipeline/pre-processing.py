## Read in libraries
import os
import csv
import pandas as pd
import numpy as np
from functools import reduce
import calendar
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
# import warnings
# warnings.filterwarnings("ignore")

# Import import helper functions
from helper_functions import cleaning
from helper_functions import scaling

## Import datasets
current_dir = os.getcwd()

# Importing steps
df_mjo = cleaning.import_mjo(current_dir)
df_nino = cleaning.import_nino(current_dir)
df_oni = cleaning.import_oni(current_dir)
df_pdo = cleaning.import_pdo(current_dir)
df_pna = cleaning.import_pna(current_dir)
df_soi1 = cleaning.import_soi1(current_dir)
df_soi2 = cleaning.import_soi2(current_dir)
df_flow = cleaning.import_flow(current_dir)
df_grace = cleaning.import_grace(current_dir)
df_snotel = cleaning.import_snotel(current_dir)

## Pre-merge cleaning steps

# Cleaning at this stage is only adjustments required to allow merging, 
# More serious cleaning follows in later stages.

# Import dictionaries used to transform month variable to numeric
from helper_functions import dictionaries

# Run functions
df_mjo = cleaning.clean_mjo(df_mjo)
df_nino = cleaning.clean_nino(df_nino)
df_oni = cleaning.clean_oni(df_oni)
df_pdo = cleaning.clean_pdo(df_pdo)
df_pna = cleaning.clean_pna(df_pna)
df_soi1 = cleaning.clean_soi1(df_soi1)
df_soi2 = cleaning.clean_soi2(df_soi2)
df_flow = cleaning.clean_flow(df_flow)
df_grace = cleaning.clean_grace(df_grace)
df_snotel = cleaning.clean_snotel(df_snotel)

# Merging dataframes
from helper_functions import merge
data_frames = [df_mjo, df_nino, df_oni, df_pdo, df_pna, df_soi1, 
                df_soi2, df_flow, df_grace, df_snotel]
df_merged = merge.merge_dfs(data_frames)

# Outlier cleaning

# Imputation

# Feature engineering from features.py

# Import functions used in scaling 
from helper_functions import scaling

# Perform preprocessing on columns except day, month, year
trans_vars = scaling.preprocess_dataframe(df_merged)

# Output the DataFrame to a CSV file
trans_vars.to_csv('transformed_vars.csv', index=False) 
