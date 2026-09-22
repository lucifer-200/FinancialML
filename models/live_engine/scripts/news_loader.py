import os
import json
import requests
import pandas as pd
from datetime import datetime, date
import pytz
from dotenv import load_dotenv

# ENV SETUP
load_dotenv()

GNEWS_KEY = os.getenv("GNEWS_API_KEY")
MARKETAUX_KEY = os.getenv("MARKETAUX_API_KEY")
NEWSDATA_KEY = os.getenv("NEWSDATA_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_API_KEY")

IST = pytz.timezone("Asia/Kolkata")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "live_engine")
DATA_DIR = os.path.join(MODEL_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

#NEW: Load stock keyword map
KEYWORD_FILE = os.path.join(MODEL_DIR, "scripts" , "stock_keywords.json")
with open(KEYWORD_FILE, "r", encoding="utf-8") as f:
    STOCK_KEYWORDS = json.load(f)


# TIME UTILS
def utc_to_ist(utc_str):
    dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
    ist_dt = dt.astimezone(IST)
    return ist_dt.date().isoformat(), ist_dt.strftime("%H:%M:%S")


# API FETCHERS (UNCHANGED)
def fetch_gnews(query, from_date, to_date):
    url = "https://gnews.io/api/v4/search"
    params = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "lang": "en",
        "country": "in",
        "max": 100,
        "apikey": GNEWS_KEY
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"GNews skipped for '{query}': {e}")
        return []

    rows = []
    for a in r.json().get("articles", []):
        d, t = utc_to_ist(a["publishedAt"])
        rows.append([d, t, a["title"], a["description"], a["source"]["name"]])

    return rows


def fetch_marketaux(query, from_date, to_date):
    url = "https://api.marketaux.com/v1/news/all"
    params = {
        "search": query, 
        "language": "en",
        "published_after": from_date,
        "published_before": to_date,
        "api_token": MARKETAUX_KEY
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"MarketAux skipped for '{query}': {e}")
        return []

    rows = []
    for a in r.json().get("data", []):
        d, t = utc_to_ist(a["published_at"])
        rows.append([d, t, a["title"], a.get("description"), a["source"]])

    return rows


def fetch_newsdata(query, from_date, to_date):
    url = "https://newsdata.io/api/1/news"
    params = {
        "apikey": NEWSDATA_KEY,
        "q": query,
        "country": "in",
        "language": "en"
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"NewsData skipped for '{query}': {e}")
        return []

    rows = []
    for a in r.json().get("results", []):
        if not a.get("pubDate"):
            continue
        d, t = utc_to_ist(a["pubDate"])
        rows.append([d, t, a["title"], a.get("description"), a.get("source_id")])

    return rows


# MAIN PIPELINE
def fetch_all_news(stock, from_date, to_date):
    today = date.today().isoformat()
    if to_date > today:
        to_date = today

    stock = stock.upper()

    #NEW: get keywords for this stock
    keywords = STOCK_KEYWORDS.get(stock, [stock])

    print(f"\nStock: {stock}")
    print(f"Using keywords: {keywords}\n")

    all_rows = []

    for kw in keywords:
        print(f"Fetching news for keyword: {kw}")

        all_rows.extend(fetch_gnews(kw, from_date, to_date))
        all_rows.extend(fetch_marketaux(kw, from_date, to_date))
        all_rows.extend(fetch_newsdata(kw, from_date, to_date))

    df = pd.DataFrame(
        all_rows,
        columns=["Date", "Time", "Headline", "Description", "Source"]
    )

    if df.empty:
        print("No news found.")
    else:
        df.drop_duplicates(subset=["Headline", "Date", "Time"], inplace=True)

    output = os.path.join(DATA_DIR, f"{stock}", f"{stock}_news.csv")
    df.to_csv(output, index=False)

    print(f"\nSaved {len(df)} news articles to {output}")


# ENTRY POINT
if __name__ == "__main__":
    stock = input("Enter stock name (e.g. TCS, ADANIENT): ")
    from_date = input("From date (YYYY-MM-DD): ")
    to_date = input("To date (YYYY-MM-DD): ")

    fetch_all_news(stock, from_date, to_date)
