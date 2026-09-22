import os
import pandas as pd

# PATH SETUP
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "live_engine", "data")


def align_news_with_ohlcv(stock: str, interval: str = "intraday"):
    stock = stock.upper()

    # File paths
    news_file = os.path.join(DATA_DIR, f"{stock}", f"{stock}_news_scored.csv")
    ohlcv_file = os.path.join(DATA_DIR, f"{stock}", f"{stock}_{interval}_ohlcv.csv")
    output_file = os.path.join(DATA_DIR, f"{stock}", f"{stock}_{interval}_lstm_input.csv")

    if not os.path.exists(news_file):
        raise FileNotFoundError(f"News file not found: {news_file}")

    if not os.path.exists(ohlcv_file):
        raise FileNotFoundError(f"OHLCV file not found: {ohlcv_file}")

    # LOAD DATA
    news_df = pd.read_csv(news_file)
    ohlcv_df = pd.read_csv(ohlcv_file)

    # TIME NORMALIZATION
    # News time: already IST but tz-naive
    news_df["news_time"] = pd.to_datetime(
        news_df["Date"] + " " + news_df["Time"],
        errors="coerce"
    )

    # OHLCV timestamp: UTC to IST to tz-naive
    ohlcv_df["timestamp"] = pd.to_datetime(
        ohlcv_df["timestamp"], utc=True, errors="coerce"
    ).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    # Sort for safety
    news_df.sort_values("news_time", inplace=True)
    ohlcv_df.sort_values("timestamp", inplace=True)

    # INIT SENTIMENT FEATURES
    ohlcv_df["signal_mean"] = 0.0
    ohlcv_df["signal_min"] = 0.0
    ohlcv_df["signal_max"] = 0.0
    ohlcv_df["signal_std"] = 0.0
    ohlcv_df["news_count"] = 0
    ohlcv_df["neg_count"] = 0

    # ALIGN NEWS TO CANDLES
    for i in range(len(ohlcv_df)):
        current_time = ohlcv_df.loc[i, "timestamp"]
        prev_time = (
            ohlcv_df.loc[i - 1, "timestamp"]
            if i > 0 else current_time
        )

        # News affecting this candle
        mask = (
            (news_df["news_time"] > prev_time) &
            (news_df["news_time"] <= current_time)
        )

        window_news = news_df.loc[mask]

        if window_news.empty:
            continue

        signals = window_news["SignalScore"]

        ohlcv_df.loc[i, "signal_mean"] = signals.mean()
        ohlcv_df.loc[i, "signal_min"] = signals.min()
        ohlcv_df.loc[i, "signal_max"] = signals.max()
        ohlcv_df.loc[i, "signal_std"] = signals.std(ddof=0)
        ohlcv_df.loc[i, "news_count"] = len(window_news)
        ohlcv_df.loc[i, "neg_count"] = (signals < 0).sum()

    # SAVE
    ohlcv_df.to_csv(output_file, index=False)
    print(f"LSTM input saved to {output_file}")


# ENTRY POINT
if __name__ == "__main__":
    stock = input("Enter stock ticker (e.g. ADANIENT): ").strip()
    interval = input("Enter interval (default: intraday): ").strip() or "intraday"

    align_news_with_ohlcv(stock, interval)
