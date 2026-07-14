import pandas as pd

# Read the CSV file
data = pd.read_csv("dataset.csv")

print("First 5 Rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nColumns:")
print(data.columns)

print("\nMissing Values:")
print(data.isnull().sum())