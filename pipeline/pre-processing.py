## Read in libraries
import os
import csv
import pandas as pd
from functools import reduce
import calendar
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
# import warnings
# warnings.filterwarnings("ignore")

## Import datasets
current_dir = os.getcwd()

# Import import functions from cleaning.py
from cleaning import import_mjo
from cleaning import import_nino
from cleaning import import_oni
from cleaning import import_pdo
from cleaning import import_pna
from cleaning import import_soi1
from cleaning import import_soi2
from cleaning import import_flow
from cleaning import import_grace
from cleaning import import_snotel

# Importing steps
df_mjo = import_mjo(current_dir)
df_nino = import_nino(current_dir)
df_oni = import_oni(current_dir)
df_pdo = import_pdo(current_dir)
df_pna = import_pna(current_dir)
df_soi1 = import_soi1(current_dir)
df_soi2 = import_soi2(current_dir)
df_flow = import_flow(current_dir)
df_grace = import_grace(current_dir)
df_snotel = import_snotel(current_dir)

## Pre-merge cleaning steps

# Cleaning at this stage is only adjustments required to allow merging, 
# More serious cleaning follows in later stages.

# Import dictionaries used to transform month variable to numeric
import dictionaries

# Import cleaning functions from cleaning.py
from cleaning import clean_mjo
from cleaning import clean_nino
from cleaning import clean_oni
from cleaning import clean_pdo
from cleaning import clean_pna
from cleaning import clean_soi1
from cleaning import clean_soi2
from cleaning import clean_flow
from cleaning import clean_grace
from cleaning import clean_snotel

# Run functions
df_mjo = clean_mjo(df_mjo)
df_nino = clean_nino(df_nino)
df_oni = clean_nino(df_oni)
df_pdo = clean_nino(df_pdo)
df_pna = clean_pna(df_pna)
df_soi1 = clean_soi1(df_soi1)
df_soi2 = clean_soi2(df_soi2)
df_flow = clean_flow(df_flow)
df_grace = clean_grace(df_grace)
df_snotel = clean_snotel(df_snotel)

## Merge code
data_frames = [df_mjo, df_nino, df_oni, df_pdo, df_pna, df_soi1, 
               df_soi2, df_flow, df_grace, df_snotel]
df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['year', 'month', 'day'],
                                            how='outer'), data_frames)

# Outlier cleaning

# Imputation

# Feature engineering from features.py

# Scaling
from scaling import preprocess_column
from scaling import preprocess_dataframe

# Perform preprocessing on columns except day, month, year
trans_vars = preprocess_dataframe(df_merged)
