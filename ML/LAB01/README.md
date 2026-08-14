# DS605: Fundamentals of Machine Learning

# Lab Assignment 1: Data Scraping and Preprocessing using Python and Scrapy

**Name:** Hitesh Rathod

**Student ID:** 202618040

**Dataset:** Books to Scrape (https://books.toscrape.com/)

---

# Project Overview

This project focuses on collecting book information from the Books to Scrape website, cleaning the collected data, creating new features, generating visualizations, and extracting meaningful insights from the dataset.

The project covers the complete data analysis pipeline:

* Data scraping
* Data preprocessing
* Feature engineering
* Data visualization
* Interpretation of results

---

# Project Files

| File                          | Description                                 |
| ----------------------------- | ------------------------------------------- |
| scrape.py                     | Scrapes book information from the website   |
| preprocess.py                 | Cleans and transforms the raw dataset       |
| visualize.py                  | Generates plots and saves them as images    |
| wordcloud_plot.py             | Creates a word cloud from book descriptions |
| raw_books.csv                 | Original scraped dataset                    |
| cleaned_books.csv             | Processed dataset                           |
| price_distribution.jpg        | Price distribution plot                     |
| rating_distribution.jpg       | Rating distribution plot                    |
| average_price_by_category.jpg | Average price by category plot              |
| price_vs_rating.jpg           | Relationship between price and rating       |
| wordcloud.jpg                 | Word cloud visualization                    |

---

# Task 1: Data Scraping

The website was scraped using Python.

The following information was collected:

* Title
* Category
* Price
* Rating
* Availability
* Product description
* UPC
* Number of reviews
* Product URL

**Total books scraped:** 100

**Pages scraped:** 5

---

# Task 2: Data Preprocessing

The following preprocessing operations were performed:

* Removed duplicate books using UPC
* Filled missing descriptions
* Converted prices into numeric values
* Converted text ratings into numerical values
* Extracted stock counts from the availability column

---

# Feature Engineering

Three additional features were created:

### 1. Description Word Count

Counts the number of words in each book description.

### 2. Price Band

Books were divided into four categories:

* Low
* Medium
* High
* Very High

### 3. Affordability Score

Calculated using:

Rating ÷ Price

This metric helps identify books that provide better value.

---

# Task 3: Visualizations

The following visualizations were generated:

### 1. Price Distribution

Shows how book prices are distributed.

### 2. Rating Distribution

Shows the number of books in each rating category.

### 3. Average Price by Category

Compares average prices across different book categories.

### 4. Price vs Rating

Examines the relationship between book prices and ratings.

### 5. Description Word Cloud

Displays the most frequently used words in book descriptions.

---

# Key Observations

1. Most books are priced between £20 and £40.

2. There is no strong relationship between price and rating.

3. Books with ratings of 3 and 5 are the most common.

4. Some categories have significantly higher average prices than others.

5. Several highly rated books are available at relatively low prices.

6. The dataset contains books from multiple categories with different pricing patterns.

---

# Data Quality

| Metric               | Result |
| -------------------- | ------ |
| Missing values       | 0      |
| Duplicate UPC values | 0      |
| Total records        | 100    |

---

# Technologies Used

* Python
* Requests
* BeautifulSoup
* Pandas
* Matplotlib
* Seaborn
* WordCloud

---

# Conclusion

The project demonstrates a complete machine-learning data preparation workflow.

The scraped dataset was cleaned, transformed, and analyzed to identify relationships between categories, prices, ratings, and availability.

The results show that higher prices do not necessarily indicate better ratings and that several categories offer good value to readers.
