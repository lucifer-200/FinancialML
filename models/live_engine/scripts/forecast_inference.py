import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "live_engine", "data")
ARTIFACT_DIR = os.path.join(BASE_DIR, "live_engine", "artifacts")
MODEL_DIR = os.path.join(BASE_DIR, "live_engine", "models")


def run_forecast_inference(stock, interval="1h"):
    stock = stock.upper()

    # ---------------- Files ----------------
    model_path = os.path.join(
        MODEL_DIR, stock, f"{stock}_{interval}_lstm_model.h5"
    )

    feature_scaler_path = os.path.join(
        ARTIFACT_DIR, stock, f"{stock}_{interval}_feature_scaler.pkl"
    )

    target_scaler_path = os.path.join(
        ARTIFACT_DIR, stock, f"{stock}_{interval}_target_scaler.pkl"
    )

    feature_path = os.path.join(
        ARTIFACT_DIR, stock, f"{stock}_{interval}_features.json"
    )

    data_path = os.path.join(
        DATA_DIR, stock, f"{stock}_{interval}_lstm_input.csv"
    )

    # ---------------- Checks ----------------
    for path in [
        model_path,
        feature_scaler_path,
        target_scaler_path,
        feature_path,
        data_path
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    # ---------------- Load artifacts ----------------
    model = load_model(model_path, compile=False)
    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)

    with open(feature_path, "r") as f:
        FEATURE_COLUMNS = json.load(f)

    df = pd.read_csv(data_path)

    # ---------------- Force numeric ----------------
    for col in FEATURE_COLUMNS + ["close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    # ---------------- Window ----------------
    WINDOW_SIZE = model.input_shape[1]

    if len(df) < WINDOW_SIZE:
        raise ValueError(
            f"Not enough data for inference. Required {WINDOW_SIZE}, found {len(df)}"
        )

    latest_window = df[FEATURE_COLUMNS].iloc[-WINDOW_SIZE:].values
    latest_window_scaled = feature_scaler.transform(latest_window)
    X_input = np.expand_dims(latest_window_scaled, axis=0)

    # ---------------- Predict ----------------
    scaled_pred = model.predict(X_input, verbose=0)[0][0]
    predicted_price = target_scaler.inverse_transform(
        [[scaled_pred]]
    )[0][0]

    last_close = float(df["close"].iloc[-1])
    delta = predicted_price - last_close
    pct_change = (delta / last_close) * 100

    # ---------------- Volatility-based range (Option A) ----------------
    returns = df["close"].pct_change().dropna()
    recent_vol = returns.tail(20).std()

    if pd.isna(recent_vol) or recent_vol == 0:
        recent_vol = 0.002  # minimal fallback volatility

    lower_bound = predicted_price * (1 - 1.5 * recent_vol)
    upper_bound = predicted_price * (1 + 1.5 * recent_vol)

    # ---------------- Interpret market signal ----------------
    signal_mean = float(df["signal_mean"].iloc[-1])
    news_count = int(df["news_count"].iloc[-1])
    neg_count = int(df["neg_count"].iloc[-1])

    if abs(pct_change) < 0.3:
        trend = "Neutral"
    elif pct_change > 0:
        trend = "Bullish"
    else:
        trend = "Bearish"

    if news_count == 0:
        volatility = "Low"
    elif neg_count > 0:
        volatility = "High"
    else:
        volatility = "Medium"

    # ---------------- Confidence score ----------------
    confidence = 75

    if recent_vol > 0.01:
        confidence -= 15
    elif recent_vol > 0.005:
        confidence -= 8

    if news_count == 0:
        confidence -= 5

    confidence = max(50, min(90, confidence))

    # ---------------- Prediction basis ----------------
    basis = [
        f"Technical Trend: {trend}",
        f"Volatility Context: {volatility}",
        "News Influence: Present" if news_count > 0 else "News Influence: None"
    ]

    # ---------------- CLI Output ----------------
    print("\nMarket Forecast Insight")
    print("------------------------")
    print(f"Stock: {stock}")
    print(f"Interval: {interval}")
    print(f"Last Close: {last_close:.2f}")
    print(f"Predicted Next Close: {predicted_price:.2f}")
    print(f"Expected Range: {lower_bound:.2f} - {upper_bound:.2f}")
    print(f"Expected Change: {pct_change:.2f}%")
    print(f"Trend: {trend}")
    print(f"Volatility Risk: {volatility}")
    print(f"Prediction Confidence: {confidence}%")

    print("\nPrediction Basis:")
    for item in basis:
        print(f"- {item}")

    # ---------------- RETURN FOR UI ----------------
    return {
        "Last Close": round(last_close, 2),
        "Predicted Close": round(predicted_price, 2),
        "Prediction Range": f"{lower_bound:.2f} - {upper_bound:.2f}",
        "Expected Change (%)": round(pct_change, 2),
        "Trend": trend,
        "Volatility Risk": volatility,
        "Confidence (%)": confidence,
        "News Count": news_count,
        "Average Signal": round(signal_mean, 3)
    }


# ================= ENTRY =================
if __name__ == "__main__":
    stock = input("Enter stock ticker (e.g. ADANIENT): ").strip()
    interval = input("Enter interval (default: 1h): ").strip() or "1h"
    run_forecast_inference(stock, interval)
