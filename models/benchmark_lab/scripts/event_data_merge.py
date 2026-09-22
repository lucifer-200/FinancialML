import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "benchmark_lab")
news_file = os.path.join(MODEL_DIR, "data", "event_news.csv")
price_file = os.path.join(MODEL_DIR, "data", "event_price_window.csv")
output_file = os.path.join(MODEL_DIR, "data", "event_merged_raw.csv")

news_frame = pd.read_csv(news_file)
price_frame = pd.read_csv(price_file)

news_frame["Date"] = pd.to_datetime(news_frame["Date"])
price_frame["Date"] = pd.to_datetime(price_frame["Date"])

news_frame = news_frame.sort_values("Date").reset_index(drop=True)
price_frame = price_frame.sort_values("Date").reset_index(drop=True)

news_frame = news_frame.drop_duplicates(subset=["Date", "Headline"])
price_frame = price_frame.drop_duplicates(subset=["Date"])
price_frame = price_frame.ffill()

daily_headlines = news_frame.groupby("Date")["Headline"].apply(lambda x: " | ".join(x)).reset_index()

merged_frame = pd.merge(price_frame, daily_headlines, on="Date", how="left")

merged_frame["Headline"] = merged_frame["Headline"].fillna("No news available")

merged_frame.to_csv(output_file, index=False)

print(f"Merging complete. File saved to {output_file}")
print("Columns in final file:")
print(merged_frame.columns)
print("\nSample data:")
print(merged_frame.head(5))
