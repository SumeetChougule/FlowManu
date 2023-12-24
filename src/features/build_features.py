import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_pickle("../../data/interim/data_processed.pkl")

df["Stage1.Output.Measurement0.U.Actual"].plot()


def clean_time_series(series, threshold=3):
    """
    Clean time series data by removing extreme values and filling missing values with linear interpolation.

    Parameters:
    - series (pd.Series): Input time series data.
    - threshold (float): Threshold for identifying extreme values using z-score. Default is 3.

    Returns:
    - pd.Series: Cleaned time series data.
    """
    # Calculate z-scores to identify extreme values
    z_scores = np.abs((series - series.mean()) / series.std())

    # Mask extreme values based on the threshold
    mask_extreme = z_scores > threshold

    # Interpolate missing values with linear interpolation
    cleaned_series = series.mask(mask_extreme).interpolate(method="linear")

    return cleaned_series


cleaned_data = clean_time_series(df["Stage1.Output.Measurement0.U.Actual"])

cleaned_data.plot()

df["Stage1.Output.Measurement0.U.Actual"] = cleaned_data

df = df.iloc[:, :42]


def feature_engineering(df, label_column, window_size):
    """
    Perform feature engineering on the input DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with the original features.
    - label_column (str): The name of the label column.

    Returns:
    - pd.DataFrame: DataFrame with additional engineered features.
    """

    # Copy the original DataFrame to avoid modifying the input directly
    df_eng = df.copy()

    # Get the list of feature columns (exclude the label column)
    feature_columns = df_eng.columns[df_eng.columns != label_column]

    # Example: Adding a moving average feature for each feature column
    window_size = window_size
    for column in feature_columns:
        df_eng[f"{column}_moving_avg"] = (
            df_eng[column].rolling(window=window_size).mean()
        )

    # Adding interaction terms based on correlation
    correlation_matrix = df_eng[feature_columns].corr()

    for i in range(len(feature_columns)):
        for j in range(i + 1, len(feature_columns)):
            col_i = feature_columns[i]
            col_j = feature_columns[j]

            # Customize the correlation threshold as needed
            if np.abs(correlation_matrix.loc[col_i, col_j]) > 0.8:
                df_eng[f"interaction_{col_i}_{col_j}"] = df_eng[col_i] * df_eng[col_j]

    # Example: Adding time-related features
    df_eng["hour_of_day"] = df_eng.index.hour
    df_eng["day_of_week"] = df_eng.index.dayofweek

    # Drop rows with NaN values
    df_eng = df_eng.dropna()

    return df_eng


# Usage
label_column = "Stage1.Output.Measurement0.U.Actual"
df_eng = feature_engineering(df, label_column, window_size=10)

df_eng.to_pickle("../../data/interim/data_engineered.pkl")
