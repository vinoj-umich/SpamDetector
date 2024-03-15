"""
Module to load data from a CSV file.
"""

import pandas as pd

def clean_data(df):
    """
    Clean the data by dropping rows with missing values and 
    converting date column to datetime format.
    
    Args:
    - df (DataFrame): The dataframe to be cleaned.
    
    Returns:
    - cleaned_df (DataFrame): The cleaned dataframe.
    """
    return df.dropna()

def load_data(csv_file):
    """
    Load data from a CSV file.

    Parameters:
        csv_file (str): Path to the CSV file.

    Returns:
        X (DataFrame): Features dataframe.
        y (Series): Target series.
    """
    df = pd.read_csv(csv_file)
    df = clean_data(df)
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return x, y
