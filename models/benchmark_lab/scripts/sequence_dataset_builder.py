import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "benchmark_lab")
DATA_DIR = os.path.join(MODEL_DIR, "data")

market_frame = pd.read_csv(os.path.join(DATA_DIR, "event_cleaned.csv"))
market_frame["Date"] = pd.to_datetime(market_frame["Date"])
market_frame = market_frame.sort_values("Date").reset_index(drop=True)

market_frame["Signal_Score"] = market_frame["Signal_Score"].fillna(0)
market_frame["Signal_Score"] = market_frame["Signal_Score"] * 0.25

num_features = ["Open", "High", "Low", "Close", "Volume"]
numeric_features = market_frame[num_features]

scaler_num = MinMaxScaler()
scaled_num = scaler_num.fit_transform(numeric_features)

scaler_signal = MinMaxScaler(feature_range=(-1, 1))
scaled_signal = scaler_signal.fit_transform(market_frame[["Signal_Score"]])

scaled_with_signal = np.concatenate([scaled_num, scaled_signal], axis=1)

def make_seq(data, seq=10):
    sequence_features, sequence_targets = [], []
    for i in range(len(data) - seq):
        sequence_features.append(data[i:i+seq])
        sequence_targets.append(data[i+seq, 3])
    return np.array(sequence_features), np.array(sequence_targets)

SEQ = 10
X_num, y_num = make_seq(scaled_num, SEQ)
X_signal, y_signal = make_seq(scaled_with_signal, SEQ)

split = int(0.8 * len(X_num))

np.save(os.path.join(DATA_DIR, "X_num_train.npy"), X_num[:split])
np.save(os.path.join(DATA_DIR, "y_num_train.npy"), y_num[:split])
np.save(os.path.join(DATA_DIR, "X_num_test.npy"),  X_num[split:])
np.save(os.path.join(DATA_DIR, "y_num_test.npy"),  y_num[split:])

np.save(os.path.join(DATA_DIR, "X_signal_train.npy"), X_signal[:split])
np.save(os.path.join(DATA_DIR, "y_signal_train.npy"), y_signal[:split])
np.save(os.path.join(DATA_DIR, "X_signal_test.npy"),  X_signal[split:])
np.save(os.path.join(DATA_DIR, "y_signal_test.npy"),  y_signal[split:])

print("Dataset prepared correctly")
