from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pandas as pd


def preprocess_column(df, column_name):
    # Skip preprocessing for specified columns
    if column_name in ['month', 'year', 'day', 'volume', 'forecast_year']:
        return df

    # Check the data type of the column
    column_dtype = df[column_name].dtype

    if column_dtype == 'object' and column_name != 'oniSEAS':
        # If it's a categorical variable, perform one-hot encoding
        df = pd.get_dummies(df, columns=[column_name], prefix=[column_name])
    elif column_dtype in ['int64', 'float64']:
        # If it's a numeric variable, standardize it
        scaler = StandardScaler()
        #df[column_name] = scaler.fit_transform(df[[column_name]])
        df[column_name] = df.groupby('site_id', index=False)[column_name].apply(lambda x: scaler.fit_transform([x]))
    elif column_dtype == 'bool':
        # If it's a binary variable, check if it's 0-1 coded and encode if not
        unique_values = df[column_name].unique()
        if set(unique_values) == {0, 1}:
            print(f"{column_name} is already 0-1 coded.")
        else:
            label_encoder = LabelEncoder()
            df[column_name] = label_encoder.fit_transform(df[column_name])
    else:
        print(f"Warning: Unsupported data type for column {column_name}")

    return df


def scale_dataframe(df):
    # Iterate over all columns in the DataFrame
    scaling_cols = [col for col in df.columns if (df[col].dtype == 'int64') or (df[col].dtype == 'float64')]

    # Now process these different dfs
    grouped = df.groupby('site_id')
    # todo fix leakage of scaling whole dataset together
    df[scaling_cols] = grouped[scaling_cols].apply(lambda x: x.fillna(x.mean())).reset_index(drop=True)
    df[scaling_cols] = (df[scaling_cols] - grouped[scaling_cols].transform('mean')) / grouped[scaling_cols].transform('std')

    return df

def scale_ground_truth(ground_truth: pd.DataFrame, gt_col: str):
    gt = ground_truth.copy()
    gt_means = gt.groupby('site_id')[gt_col].mean().to_dict()
    gt_stds = gt.groupby('site_id')[gt_col].std().to_dict()
    gt['gt_mean'] = gt['site_id'].map(gt_means)
    gt['gt_std'] = gt['site_id'].map(gt_stds)
    gt[gt_col] = gt[gt_col] - gt['gt_mean']
    gt[gt_col] = gt[gt_col] / gt['gt_std']
    gt.drop(columns=['gt_mean', 'gt_std'], inplace=True)
    return gt, gt_means, gt_stds

