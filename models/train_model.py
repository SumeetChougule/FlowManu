import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_pickle("../../data/processed/best_features.pkl")

df.info()


def time_series_train_test_split(X, y, test_size=0.2):
    """
    Perform time series aware train-test split.

    Parameters:
    - X: Features DataFrame
    - y: Target Series
    - test_size: Proportion of the data to include in the test split

    Returns:
    - X_train, X_test, y_train, y_test
    """
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Train and evaluate a model.

    Parameters:
    - model: Machine learning model
    - X_train, X_test: Training and testing features
    - y_train, y_test: Training and testing target

    Returns:
    - mean absolute error, r2 score
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, r2, y_test, y_pred


def visualize_results(y_test, y_pred, model_name):
    """
    Visualize the actual vs predicted results.

    Parameters:
    - y_test: Actual target values
    - y_pred: Predicted target values
    - model_name: Name of the model for the plot title
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label="Actual")
    plt.plot(y_test.index, y_pred, label="Predicted")
    plt.title(f"{model_name} - Actual vs Predicted")
    plt.xlabel("Time")
    plt.ylabel("Stage1.Output.Measurement0.U.Actual")
    plt.legend()
    plt.show()


def experiment_with_models(df):
    """
    Experiment with different models and visualize the results.

    Parameters:
    - df: Input DataFrame containing features and target

    Returns:
    - None
    """
    target_column = "Stage1.Output.Measurement0.U.Actual"
    features = df.drop(target_column, axis=1)
    target = df[target_column]

    # Scale the features
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features), columns=features.columns
    )

    models = {
        "RandomForest": RandomForestRegressor(),
        "LinearRegression": LinearRegression(),
        "KNeighbors": KNeighborsRegressor(),
    }

    for model_name, model in models.items():
        X_train, X_test, y_train, y_test = time_series_train_test_split(
            features_scaled, target
        )
        mae, r2, y_test_pred, y_pred = evaluate_model(
            model, X_train, X_test, y_train, y_test
        )
        print(f"{model_name}: MAE = {mae:.2f}, R2 Score = {r2:.2f}")
        visualize_results(
            pd.Series(y_test_pred, index=y_test.index), y_pred, model_name
        )


# Assuming df is your DataFrame
experiment_with_models(df)
