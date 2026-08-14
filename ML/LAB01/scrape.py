import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

session = requests.Session()

books = []

for page in range(1, 6):

    url = f"https://books.toscrape.com/catalogue/page-{page}.html"

    print(f"Scraping page {page}...")

    response = session.get(url, timeout=10)

    soup = BeautifulSoup(response.text, "html.parser")

    for book in soup.select("article.product_pod"):

        book_url = urljoin(
            "https://books.toscrape.com/catalogue/",
            book.h3.a["href"],
        )

        book_page = session.get(book_url, timeout=10)

        book_soup = BeautifulSoup(book_page.text, "html.parser")

        table = {
            row.th.text: row.td.text
            for row in book_soup.select("table tr")
        }

        description = book_soup.select_one(
            "#product_description + p"
        )

        books.append(
            {
                "title": book_soup.find("h1").text,
                "category": book_soup.select(
                    "ul.breadcrumb li"
                )[2].text.strip(),
                "price": book_soup.select_one(
                    ".price_color"
                ).text,
                "rating": book_soup.select_one(
                    ".star-rating"
                )["class"][1],
                "availability": book_soup.select_one(
                    ".availability"
                ).text.strip(),
                "description": (
                    description.text if description else ""
                ),
                "upc": table.get("UPC"),
                "number_of_reviews": table.get(
                    "Number of reviews"
                ),
                "product_url": book_url,
            }
        )

df = pd.DataFrame(books)

df.to_csv(
    "raw_books.csv",
    index=False
)

print(df.shape)
print("Finished.")
