import os
import pandas as pd
import matplotlib.pyplot as plt


def show_charts(stock, interval):
    stock = stock.upper()

    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")

    price_file = os.path.join(
        data_dir, f"{stock}", f"{stock}_{interval}_ohlcv.csv"
    )
    lstm_file = os.path.join(
        data_dir, f"{stock}", f"{stock}_{interval}_lstm_input.csv"
    )

    if not os.path.exists(price_file):
        raise FileNotFoundError(price_file)

    if not os.path.exists(lstm_file):
        raise FileNotFoundError(lstm_file)

    price_df = pd.read_csv(price_file)
    sent_df = pd.read_csv(lstm_file)

    price_df["timestamp_ist"] = pd.to_datetime(
        price_df["timestamp_ist"], errors="coerce"
    )
    sent_df["timestamp_ist"] = pd.to_datetime(
        sent_df["timestamp_ist"], errors="coerce"
    )

    price_df.dropna(inplace=True)
    sent_df.dropna(inplace=True)

    # Window 1: Price
    plt.figure(figsize=(11, 5))
    plt.plot(
        price_df["timestamp_ist"],
        price_df["close"],
        linewidth=2
    )
    plt.title(f"{stock} Price Movement ({interval})")
    plt.xlabel("Time")
    plt.ylabel("Close Price")
    plt.grid(alpha=0.3)

    # Window 2: Sentiment
    plt.figure(figsize=(11, 4))
    plt.plot(
        sent_df["timestamp_ist"],
        sent_df["signal_mean"],
        linewidth=2,
        label="Average Signal"
    )
    plt.bar(
        sent_df["timestamp_ist"],
        sent_df["news_count"],
        alpha=0.3,
        label="News Count"
    )
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(f"{stock} Signal & News Impact ({interval})")
    plt.xlabel("Time")
    plt.ylabel("Signal / News Volume")
    plt.legend()
    plt.grid(alpha=0.3)

    # KEEP WINDOWS OPEN
    plt.show()
