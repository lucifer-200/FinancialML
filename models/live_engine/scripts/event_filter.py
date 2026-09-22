import os
import json
import pandas as pd
import re

# PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "live_engine")
DATA_DIR = os.path.join(MODEL_DIR, "data")
KEYWORD_FILE = os.path.join(MODEL_DIR, "scripts", "stock_keywords.json")
EVENT_KEYWORDS_FILE = os.path.join(MODEL_DIR, "scripts", "event_keywords.json")

# EVENT DEFINITIONS
EVENT_KEYWORDS = json.load(open(EVENT_KEYWORDS_FILE, "r", encoding="utf-8"))

# UTILS
def normalize(text):
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", text.lower())


def detect_event_type(text):
    for event, words in EVENT_KEYWORDS.items():
        for w in words:
            if w in text:
                return event
    return "GENERAL"


def compute_relevance_score(text, company_keywords):
    score = 0

    # company/entity mention
    for kw in company_keywords:
        if kw.lower() in text:
            score += 2

    # event impact
    for words in EVENT_KEYWORDS.values():
        for w in words:
            if w in text:
                score += 3

    return score


def is_relevant(row, company_keywords, threshold=3):
    text = normalize(f"{row['Headline']} {row['Description']}")
    score = compute_relevance_score(text, company_keywords)
    event = detect_event_type(text)
    return score >= threshold, score, event


# MAIN FILTER
def filter_news_for_stock(stock):
    stock = stock.upper()

    # Load keyword map
    with open(KEYWORD_FILE, "r", encoding="utf-8") as f:
        stock_keywords = json.load(f)

    if stock not in stock_keywords:
        raise ValueError(f"No keywords found for stock: {stock}")

    company_keywords = stock_keywords[stock]

    input_file = os.path.join(DATA_DIR, f"{stock}", f"{stock}_news.csv")
    output_file = os.path.join(DATA_DIR, f"{stock}", f"{stock}_news_filtered.csv")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"News file not found: {input_file}")

    df = pd.read_csv(input_file)

    filtered_rows = []

    for _, row in df.iterrows():
        relevant, score, event_type = is_relevant(row, company_keywords)
        if relevant:
            row["RelevanceScore"] = score
            row["EventType"] = event_type
            filtered_rows.append(row)

    filtered_df = pd.DataFrame(filtered_rows)
    filtered_df.to_csv(output_file, index=False)

    print(f"{stock}: kept {len(filtered_df)} / {len(df)} articles")
    print(f"Saved to {output_file}")


# ENTRY POINT
if __name__ == "__main__":
    stock = input("Enter stock ticker (e.g. ADANIENT): ").strip()
    filter_news_for_stock(stock)
