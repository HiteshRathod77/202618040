import pandas as pd
import matplotlib.pyplot as plt


# Load cleaned data
df = pd.read_csv("cleaned_books.csv")


# Calculate average price for each category
category_price = (
    df.groupby("category")["price"]
    .mean()
    .sort_values(ascending=False)
)


# Create bar chart
category_price.plot(kind="bar")

plt.xlabel("Category")
plt.ylabel("Average Price (£)")
plt.title("Average Book Price by Category")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.show()
