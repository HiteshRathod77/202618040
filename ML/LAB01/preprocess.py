import pandas as pd


# Load raw dataset
df = pd.read_csv("raw_books.csv")


# -----------------------------
# 1. Clean price
# -----------------------------

df["price"] = (
    df["price"]
    .str.replace("£", "", regex=False)
    .astype(float)
)


# -----------------------------
# 2. Convert rating to numbers
# -----------------------------

rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

df["rating"] = df["rating"].str.replace(
    "star-rating ", "", regex=False
)

df["rating"] = df["rating"].map(rating_map)


# -----------------------------
# 3. Extract stock count
# -----------------------------

df["stock_count"] = df["availability"].str.extract(
    r"(\d+)"
).astype(int)


# -----------------------------
# 4. Clean description
# -----------------------------

df["description"] = df["description"].fillna(
    "No description available"
)


# -----------------------------
# Check results
# -----------------------------

print("\nCleaned data:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nStock count:")
print(df["stock_count"].head())

print("\nRating:")
print(df["rating"].head())

print("\nPrice:")
print(df["price"].head())


#2 hitesh

# -----------------------------
# 5. Feature Engineering
# -----------------------------

# Feature 1: Number of words in description
df["description_word_count"] = (
    df["description"]
    .str.split()
    .str.len()
)


# Feature 2: Price band
df["price_band"] = pd.cut(
    df["price"],
    bins=[0, 20, 40, 60, 100],
    labels=["Low", "Medium", "High", "Very High"]
)


# Feature 3: Affordability score
df["affordability_score"] = (
    df["rating"] / df["price"]
)


# Display new features
print("\nNew Features:")

print(
    df[
        [
            "title",
            "description_word_count",
            "price_band",
            "affordability_score"
        ]
    ].head()
)

#3 hitesh

# Save cleaned dataset
df.to_csv("cleaned_books.csv", index=False)

print("\nCleaned dataset saved as cleaned_books.csv")

print("\nAverage book price:")
print(df["price"].mean())


print("\nAverage price by category:")

category_prices = df.groupby("category")["price"].mean()

print(category_prices)


print("\nMost expensive book:")

most_expensive = df.loc[df["price"].idxmax()]

print(most_expensive[["title", "price", "category"]])


print("\nCheapest book:")

cheapest = df.loc[df["price"].idxmin()]

print(cheapest[["title", "price", "category"]])


print("\nNumber of books by rating:")

rating_counts = df["rating"].value_counts().sort_index()

print(rating_counts)


print("\nAverage price by rating:")

rating_prices = df.groupby("rating")["price"].mean()

print(rating_prices)

print("\nCorrelation between rating and price:")

correlation = df["rating"].corr(df["price"])

print(correlation)
