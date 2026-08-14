import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_books.csv")

text = " ".join(df["description"].dropna())

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(text)

plt.figure(figsize=(12, 6))

plt.imshow(wordcloud)

plt.axis("off")

plt.title("Word Cloud of Book Descriptions")

plt.show()
