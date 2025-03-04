import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def dataset_summary(df: pd.DataFrame) -> None:
    """
    Print a summary of the dataset, including the number of rows, columns, and data types.
    """
    print("\nStatistics distribution of the data :")
    print(df.describe())
    print("\nColumn Data Types:")
    print(df.dtypes)
    print("\nFirst few rows of the dataset:")
    print(df.head())

def analyze_dataset(file_path: str) -> None:
    """
    Load a dataset and display its structure.
    """
    print(f"Loading dataset from: {file_path}")
    try:
        df = load_dataset(file_path)
        dataset_summary(df)
    except Exception as e:
        print(f"An error occurred: {e}")
