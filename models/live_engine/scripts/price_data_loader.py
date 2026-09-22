import os
import yfinance as yf
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import warnings
from pandas.errors import PerformanceWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=PerformanceWarning)


load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "live_engine")
DATA_DIR = os.path.join(MODEL_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def download_stock_data(symbol, start_date, end_date, interval="1h"):
    print(f"Downloading {interval} data for {symbol}...")

    os.makedirs(os.path.join(DATA_DIR, f"{symbol}"), exist_ok=True)
    os.makedirs(os.path.join(MODEL_DIR, "models", f"{symbol}"), exist_ok=True)
    os.makedirs(os.path.join(MODEL_DIR, "artifacts", f"{symbol}"), exist_ok=True)

    stock = symbol.upper()

    if not symbol.endswith(".NS"):
        symbol = symbol + ".NS"

    data = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        print("No data returned")
        return

    # Reset index to expose datetime
    data.reset_index(inplace=True)

    # Rename columns
    data.rename(columns={
        "Datetime": "timestamp",
        "Date": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    }, inplace=True)

    # Drop Adj Close if present
    if "Adj Close" in data.columns:
        data.drop(columns=["Adj Close"], inplace=True)

    # Convert timestamps
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data["timestamp_ist"] = data["timestamp"].dt.tz_convert("Asia/Kolkata")

    # Keep only NSE market hours
    data = data[
        (data["timestamp_ist"].dt.time >= datetime.strptime("09:15", "%H:%M").time()) &
        (data["timestamp_ist"].dt.time <= datetime.strptime("15:30", "%H:%M").time())
    ]

    # Sort by time (CRITICAL for LSTM)
    data.sort_values("timestamp_ist", inplace=True)

    # Fix zero volume (optional but recommended)
    median_volume = data["volume"].median()
    data["volume"] = data["volume"].replace(0, median_volume)

    # Final column order
    data = data[
        ["timestamp", "timestamp_ist", "open", "high", "low", "close", "volume"]
    ]

    output_file = os.path.join(DATA_DIR, f"{stock}", f"{stock}_{interval}_ohlcv.csv")

    data.to_csv(output_file, index=False)
    print(f"Saved clean intraday data to {output_file}")


if __name__ == "__main__":
    download_stock_data(
        symbol="ADANIENT.NS",
        start_date="2026-01-01",
        end_date="2026-01-04",
        interval="1h"
    )
