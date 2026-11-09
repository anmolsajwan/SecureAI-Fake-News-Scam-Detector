
import pandas as pd

# Load fake news dataset
df = pd.read_csv("data/fake_news.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Display class distribution
print("\nClass Distribution:")
print(df["label"].value_counts())
