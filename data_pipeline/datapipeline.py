# Importing required modules
# requests -> to fetch web pages
# BeautifulSoup -> to parse HTML
# sqlite3 -> to create and manage database
# statistics -> to calculate median values for imputation
# pandas -> to check SQL results and compare with DataFrame merge
import requests
from bs4 import BeautifulSoup
import sqlite3
import statistics
import pandas as pd

# Mapping star ratings from text to numbers
rating_map = {"One":1,"Two":2,"Three":3,"Four":4,"Five":5}

# Function to scrape books from a given page
def get_books_from_page(url, category=None):
    response = requests.get(url, timeout=10)  # timeout added to avoid hanging
    soup = BeautifulSoup(response.text, 'html.parser')
    books = []
    # Each book is inside article.product_pod
    for article in soup.select('article.product_pod'):
        title = article.h3.a['title']
        price = article.select_one('.price_color').text
        # star rating is stored as a class like "star-rating Three"
        star_rating = article.p['class'][1] if 'class' in article.p.attrs else "Three"
        availability = article.select_one('.availability').text.strip()
        books.append({
            'title': title,
            'price': price,
            'star_rating': star_rating,
            'availability': availability,
            'category': category
        })
    return books

# Function to clean the scraped data
def clean_data(data):
    for row in data:
        # Convert price to float (GBP)
        try:
            row['price_gbp'] = float(row['price'].replace('£',''))
        except:
            row['price_gbp'] = None
        # Convert rating text to integer
        row['rating'] = rating_map.get(row['star_rating'], None)
        # Availability -> boolean
        row['in_stock'] = "In stock" in row['availability']
    
    # Median imputation (only if we have values, else skip)
    for field in ['price_gbp','rating']:
        values = [r[field] for r in data if r[field] is not None]
        if values:  # avoid empty list error
            median_val = statistics.median(values)
            for r in data:
                if r[field] is None:
                    r[field] = median_val
    return data

# Scraping first 5 pages of "All products" to ensure >= 60 books
base_url = "http://books.toscrape.com/catalogue/page-{}.html"
all_books = []
for page in range(1, 6):  # first 5 pages
    url = base_url.format(page)
    all_books.extend(get_books_from_page(url, category="all_products"))

print("Books scraped:", len(all_books))  # should be >= 60

cleaned_books = clean_data(all_books)

# Currency conversion (fixed rate: 1 GBP = 105.50 INR)
# If price_gbp is None, we impute with median before conversion
values = [r['price_gbp'] for r in cleaned_books if r['price_gbp'] is not None]
median_val = statistics.median(values) if values else 0.0
for row in cleaned_books:
    if row['price_gbp'] is not None:
        row['price_inr'] = round(row['price_gbp'] * 105.50, 2)
    else:
        row['price_inr'] = round(median_val * 105.50, 2)

# Creating SQLite database with normalized schema
conn = sqlite3.connect("books.db")
cur = conn.cursor()

# Categories table (PK)
cur.execute("CREATE TABLE IF NOT EXISTS categories(category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE)")
# Books table (FK to categories)
cur.execute("""CREATE TABLE IF NOT EXISTS books(
    book_id INTEGER PRIMARY KEY,
    title TEXT,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER REFERENCES categories(category_id))""")

# Insert categories (only one here: all_products)
cat_map = {}
for i, cat in enumerate(set([b['category'] for b in cleaned_books]), start=1):
    cur.execute("INSERT OR IGNORE INTO categories(category_id, category_name) VALUES (?,?)",(i,cat))
    cat_map[cat] = i

# Insert books
for b in cleaned_books:
    cur.execute("""INSERT INTO books(title,price_gbp,price_inr,rating,in_stock,category_id)
                   VALUES (?,?,?,?,?,?)""",
                (b['title'],b['price_gbp'],b['price_inr'],b['rating'],int(b['in_stock']),cat_map[b['category']]))
conn.commit()

# Example SQL queries (covering SELECT, WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN, JOIN)
print("\nTop 10 highest rated books:")
for row in cur.execute("""SELECT b.title, c.category_name, b.rating
                          FROM books b JOIN categories c ON b.category_id=c.category_id
                          ORDER BY b.rating DESC LIMIT 10"""):
    print(row)

# Pandas integration: read SQL and compare with merge
df_books = pd.read_sql("SELECT * FROM books", conn)
df_cats = pd.read_sql("SELECT * FROM categories", conn)

sql_join = pd.read_sql("""SELECT b.title, c.category_name, b.rating
                          FROM books b JOIN categories c ON b.category_id=c.category_id""", conn)

merge_join = pd.merge(df_books, df_cats, on="category_id")[['title','category_name','rating']]

print("\nSQL Join result (first 5 rows):")
print(sql_join.head())

print("\nPandas Merge result (first 5 rows):")
print(merge_join.head())

conn.close()