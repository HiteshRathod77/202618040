import pandas as pd

df = pd.read_csv("cleaned_books.csv")

print("Average book price:")
print(df["price"].mean())

print("\nMost expensive book:")
print(df.loc[df["price"].idxmax(), ["title", "price", "category"]])

print("\nCheapest book:")
print(df.loc[df["price"].idxmin(), ["title", "price", "category"]])

print("\nNumber of books by rating:")
print(df["rating"].value_counts().sort_index())

print("\nAverage price by rating:")
print(df.groupby("rating")["price"].mean())

print("\nCorrelation between rating and price:")
print(df["rating"].corr(df["price"]))
