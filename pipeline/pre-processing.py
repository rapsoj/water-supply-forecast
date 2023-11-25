# Read in libraries
import os
import csv
import pandas as pd
from functools import reduce
import calendar
from datetime import datetime
# import warnings
# warnings.filterwarnings("ignore")

current_dir = os.getcwd()

import dictionaries

# Import import and cleaning functions for each dataset from cleaning.py
from cleaning import import_mjo
from cleaning import clean_mjo
from cleaning import import_nino
from cleaning import clean_nino
from cleaning import import_oni
from cleaning import clean_oni
from cleaning import import_pdo
from cleaning import clean_pdo
from cleaning import import_pna
from cleaning import clean_pna
from cleaning import import_soi1
from cleaning import clean_soi1
from cleaning import import_soi2
from cleaning import clean_soi2
from cleaning import import_flow
from cleaning import clean_flow
from cleaning import import_grace
from cleaning import clean_grace
from cleaning import import_snotel
from cleaning import clean_snotel

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

# Pre-merge cleaning steps
# Cleaning at this stage is only adjustments required to allow merging, 
# More serious cleaning follows in later stages.
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

# Merging functions 

# Outlier cleaning

# Imputation

# Feature engineering from features.py

