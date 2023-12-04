from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pandas as pd


def preprocess_column(df, column_name):
    # Skip preprocessing for specified columns
    if column_name in ['month', 'year', 'day', 'volume', 'forecast_year']:
        return df

    # Check the data type of the column
    column_dtype = df[column_name].dtype

    if column_dtype == 'object':
        # If it's a categorical variable, perform one-hot encoding
        df = pd.get_dummies(df, columns=[column_name], prefix=[column_name])
    elif column_dtype in ['int64', 'float64']:
        # If it's a numeric variable, standardize it
        scaler = StandardScaler()
        df[column_name] = scaler.fit_transform(df[[column_name]])
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


def preprocess_dataframe(df):
    # Iterate over all columns in the DataFrame
    for column_name in df.columns:
        df = preprocess_column(df, column_name)

    return df
