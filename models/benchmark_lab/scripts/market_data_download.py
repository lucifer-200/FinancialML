import yfinance as yf
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "benchmark_lab")
DATA_PATH = os.path.join(MODEL_DIR, "data", "event_price_window.csv")
DEFAULT_SYMBOL = "ADANIENT.NS"
START_DATE = "2022-12-01"
END_DATE = "2023-03-31"

print(f"Data path set to: {DATA_PATH}")

market_data = yf.download(DEFAULT_SYMBOL, start=START_DATE, end=END_DATE)
market_data.reset_index(inplace=True)
market_data.to_csv(DATA_PATH, index=False)
print(f"Stock data for {DEFAULT_SYMBOL} downloaded and saved to {DATA_PATH}")
