import numpy as np
import pandas as pd
import matplotlib as plt

df = pd.read_csv("../../data/raw/continuous_factory_process.csv")


def preprocess_dataframe(df):
    """
    Preprocess a DataFrame by selecting specific columns, removing those with 'setpoint' in the name,
    converting 'time_stamp' to datetime, and setting it as the index.

    Parameters:
    - df: pandas.DataFrame, the input DataFrame.

    Returns:
    - pandas.DataFrame, the preprocessed DataFrame.
    """
    # Select the first 72 columns
    selected_columns = df.columns[:72]
    df = df[selected_columns]

    # Remove columns containing 'setpoint' in their name
    df = df.loc[:, ~df.columns.str.contains("setpoint", case=False)]

    # Convert 'time_stamp' to datetime and set it as the index
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    df = df.set_index("time_stamp")

    return df


df_processed = preprocess_dataframe(df)

df_processed.info()


df_processed.to_pickle("../../data/interim/data_processed.pkl")
