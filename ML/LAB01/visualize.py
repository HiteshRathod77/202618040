import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load cleaned data
df = pd.read_csv("cleaned_books.csv")

# -------------------------
# 1. Price distribution
# -------------------------

plt.figure(figsize=(8, 5))

plt.hist(df["price"], bins=10)

plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Number of Books")

plt.savefig("price_distribution.png")

plt.show()


# -------------------------
# 2. Rating distribution
# -------------------------

plt.figure(figsize=(8, 5))

df["rating"].value_counts().sort_index().plot(kind="bar")

plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Books")

plt.savefig("rating_distribution.png")

plt.show()


# -------------------------
# 3. Average price by category
# -------------------------

plt.figure(figsize=(10, 6))

(
    df.groupby("category")["price"]
    .mean()
    .sort_values()
    .plot(kind="bar")
)

plt.title("Average Price by Category")
plt.xlabel("Category")
plt.ylabel("Average Price")

plt.xticks(rotation=45)

plt.savefig("average_price_by_category.png")

plt.show()


# -------------------------
# 4. Price vs Rating
# -------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["rating"],
    df["price"]
)

plt.title("Price vs Rating")
plt.xlabel("Rating")
plt.ylabel("Price")

plt.savefig("price_vs_rating.png")

plt.show()


# -------------------------
# 5. Word cloud
# -------------------------

text = " ".join(df["description"])

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(text)

plt.figure(figsize=(10, 5))

plt.imshow(wordcloud)

plt.axis("off")

plt.title("Book Description Word Cloud")

plt.savefig("wordcloud.png")

plt.show()


# -------------------------
# Analysis
# -------------------------

print("\nAverage price:")
print(df["price"].mean())

print("\nMost expensive category:")
print(
    df.groupby("category")["price"]
    .mean()
    .sort_values(ascending=False)
    .head()
)

print("\nBooks by rating:")
print(
    df["rating"]
    .value_counts()
    .sort_index()
)

print("\nPrice and rating correlation:")
print(
    df["price"]
    .corr(df["rating"])
)
