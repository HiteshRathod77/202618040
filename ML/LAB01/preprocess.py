import pandas as pd
import numpy as np
import re

# Load dataset
df = pd.read_csv("raw_books.csv")

# Remove duplicate books
df = df.drop_duplicates(subset="upc")

# Clean text columns
text_columns = [
    "title",
    "category",
    "availability",
    "description"
]

for column in text_columns:
    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

# Handle missing descriptions
df["description"] = df["description"].replace(
    "",
    "No description available"
)

# Convert price to numeric
df["price"] = (
    df["price"]
    .astype(str)
    .str.extract(r"(\d+\.\d+)")[0]
    .astype(float)
)

# Convert ratings to numbers
rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

df["rating"] = df["rating"].map(rating_map)

# Extract stock count
df["stock_count"] = (
    df["availability"]
    .str.extract(r"(\d+)")[0]
    .fillna(0)
    .astype(int)
)

# Feature 1
df["description_word_count"] = (
    df["description"]
    .str.split()
    .str.len()
)

# Feature 2
df["price_band"] = pd.cut(
    df["price"],
    bins=[0, 20, 40, 60, 100],
    labels=[
        "Low",
        "Medium",
        "High",
        "Very High"
    ]
)

# Feature 3
df["affordability_score"] = (
    df["rating"] / df["price"]
)

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Duplicate UPC values
print("\nDuplicate UPC values:")
print(df["upc"].duplicated().sum())

# Total records
print("\nTotal records:")
print(len(df))

# Save cleaned dataset
df.to_csv(
    "cleaned_books.csv",
    index=False
)

print("\nCleaned dataset saved.")

print(df.head())
