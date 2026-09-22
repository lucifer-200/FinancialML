import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# PATH SETUP
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "live_engine", "data")
ARTIFACT_DIR = os.path.join(BASE_DIR, "live_engine", "artifacts")

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# CONFIG
FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "signal_mean", "signal_min", "signal_max", "signal_std",
    "news_count", "neg_count"
]

TARGET_COLUMN = "close"
WINDOW_SIZE = 5


def build_lstm_dataset(stock, interval="1h"):
    stock = stock.upper()

    input_file = os.path.join(
        DATA_DIR, f"{stock}", f"{stock}_{interval}_lstm_input.csv"
    )

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)

    # FORCE NUMERIC
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    if len(df) <= WINDOW_SIZE:
        raise ValueError("Not enough rows to create sequences")

    # FEATURE + TARGET
    X_raw = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_raw = df[[TARGET_COLUMN]].to_numpy(dtype=np.float32)

    # SCALERS
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_scaled = feature_scaler.fit_transform(X_raw)
    y_scaled = target_scaler.fit_transform(y_raw)

    # SEQUENCES
    X, y = [], []

    for i in range(WINDOW_SIZE, len(X_scaled)):
        X.append(X_scaled[i - WINDOW_SIZE:i])
        y.append(y_scaled[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # SAVE
    np.save(os.path.join(ARTIFACT_DIR, f"{stock}", f"{stock}_{interval}_X.npy"), X)
    np.save(os.path.join(ARTIFACT_DIR, f"{stock}", f"{stock}_{interval}_y.npy"), y)

    joblib.dump(
        feature_scaler,
        os.path.join(ARTIFACT_DIR, f"{stock}", f"{stock}_{interval}_feature_scaler.pkl")
    )

    joblib.dump(
        target_scaler,
        os.path.join(ARTIFACT_DIR, f"{stock}", f"{stock}_{interval}_target_scaler.pkl")
    )

    with open(
        os.path.join(ARTIFACT_DIR, f"{stock}", f"{stock}_{interval}_features.json"),
        "w"
    ) as f:
        json.dump(FEATURE_COLUMNS, f, indent=4)

    print("Data preparation completed")
    print("X shape:", X.shape)
    print("y shape:", y.shape)


if __name__ == "__main__":
    stock = input("Enter stock ticker (e.g. ADANIENT): ").strip()
    interval = input("Enter interval (default: 1h): ").strip() or "1h"

    build_lstm_dataset(stock, interval)
