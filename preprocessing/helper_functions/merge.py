import os
import csv
import pandas as pd
import numpy as np
from functools import reduce


def merge_site_id_day(list_dataframes):
    # Merge on site id, day
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['year', 'month', 'day', 'site_id'],
                                                    how='outer'), list_dataframes)
    return df_merged


def merge_day(list_dataframes):
    # Merge on site id, day
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['year', 'month', 'day'],
                                                    how='outer'), list_dataframes)
    return df_merged
