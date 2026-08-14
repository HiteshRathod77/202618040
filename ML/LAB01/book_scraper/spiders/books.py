import scrapy


class BooksSpider(scrapy.Spider):

    name = "books"

    allowed_domains = ["books.toscrape.com"]

    start_urls = [
        "https://books.toscrape.com/catalogue/page-1.html"
    ]

    # We need at least 5 pages
    max_pages = 5

    def parse(self, response):

        print("PAGE:", response.url)

        # Find all book links on the current page
        books = response.css(
            "article.product_pod h3 a::attr(href)"
        ).getall()

        print("BOOKS FOUND:", len(books))

        # Visit every book
        for book_url in books:

            yield response.follow(
                book_url,
                callback=self.parse_book
            )

        # Find the next page
        next_page = response.css(
            "li.next a::attr(href)"
        ).get()

        if next_page:

            # Get current page number
            current_page = int(
                response.url.split("page-")[1].split(".html")[0]
            )

            # Continue only until page 5
            if current_page < self.max_pages:

                yield response.follow(
                    next_page,
                    callback=self.parse
                )

    def parse_book(self, response):

        yield {

            "title": response.css(
                "div.product_main h1::text"
            ).get(),

            "price": response.css(
                "div.product_main p.price_color::text"
            ).get(),

            "rating": response.css(
                "div.product_main p.star-rating::attr(class)"
            ).get(),

            "availability": " ".join(
                response.css(
                    "div.product_main p.instock.availability::text"
                ).getall()
            ).strip(),

            "category": response.css(
                "ul.breadcrumb li:nth-child(3) a::text"
            ).get(),

            "description": response.css(
                "#product_description ~ p::text"
            ).get(),

            "upc": response.xpath(
                "//th[normalize-space()='UPC']/following-sibling::td/text()"
            ).get(),

            "number_of_reviews": response.xpath(
                "//th[normalize-space()='Number of reviews']/following-sibling::td/text()"
            ).get(),

            "product_url": response.url
        }
