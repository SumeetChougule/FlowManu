import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_pickle("../../data/interim/data_processed.pkl")

df.info()

# Group the columns based on their properties
ambient_columns = [col for col in df.columns if col.startswith("AmbientConditions")]
machine_columns = [col for col in df.columns if col.startswith("Machine")]
combiner_columns = [col for col in df.columns if col.startswith("FirstStage")]
stage_output_columns = [col for col in df.columns if col.startswith("Stage")]


# Create a function to plot line plots for a group of columns
def plot_columns(columns, title):
    plt.figure(figsize=(20, 5))
    for col in columns:
        sns.lineplot(data=df, x=df.index, y=col, label=col)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(loc="best")
    plt.show()


# Function to extract properties from machine columns
def get_machine_properties(machine_columns):
    properties = set()
    for col in machine_columns:
        prop = ".".join(col.split(".")[1:])
        properties.add(prop)
    return properties


machine_properties = get_machine_properties(machine_columns)


# Function to create line plots for each property across all machines
def plot_machine_columns(properties, title):
    for prop in properties:
        plt.figure(figsize=(15, 6))
        for machine in range(1, 4):
            col = f"Machine{machine}.{prop}"
            sns.lineplot(data=df, x=df.index, y=col, label=col)
        plt.title(f"{title}:{prop}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend(loc="best")
        plt.show()


# plot line plots for each group of columns
plot_columns(ambient_columns, "Ambient Conditions")
plot_machine_columns(machine_properties, "Machine Properties")
plot_columns(combiner_columns, "First Stage Combiner Operation")
plot_columns(stage_output_columns, "Stage 1 Output Measurements")


# Function to plot line plots for a group of columns
def plot_columns(columns, title):
    for col in columns:
        plt.figure(figsize=(20, 5))
        sns.lineplot(data=df, x=df.index, y=col, label=col)
        plt.title(f"{title}: {col}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend(loc="best")
        plt.show()


# plot line plots for each group of columns
plot_columns(ambient_columns, "Ambient Conditions")
plot_columns(machine_properties, "Machine Properties")

# Separate plots for each column in stage_output_columns
for col in stage_output_columns:
    plot_columns([col], "Stage 1 Output Measurements")
