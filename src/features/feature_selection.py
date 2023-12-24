import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression

df = pd.read_pickle("../../data/interim/data_engineered.pkl")


def select_best_features(df, target_column, k=10):
    """
    Select the top k features based on the F-statistic between features and the target variable.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing features and target.
    - target_column (str): The name of the target variable.
    - k (int): Number of top features to select.

    Returns:
    - pd.DataFrame: DataFrame containing only the selected features.
    """

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Use F-statistic for feature selection
    selector = SelectKBest(f_regression, k=k)
    X_selected = selector.fit_transform(X, y)

    # Get the indices of the selected features
    selected_indices = selector.get_support(indices=True)

    # Get the names of the selected features
    selected_features = X.columns[selected_indices]

    # Add the target column back to the selected features
    selected_features = list(selected_features) + [target_column]

    # Create a DataFrame with only the selected features
    df_selected = df[selected_features]

    return df_selected


target_column = "Stage1.Output.Measurement0.U.Actual"
best_features_df = select_best_features(df, target_column, k=50)

best_features_df.columns

best_features_df.to_pickle("../../data/processed/best_features.pkl")
