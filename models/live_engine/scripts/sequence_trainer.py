import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# PATH SETUP
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "live_engine", "artifacts")
MODEL_DIR = os.path.join(BASE_DIR, "live_engine", "models")

os.makedirs(MODEL_DIR, exist_ok=True)


def train_sequence_model(stock, interval="1h"):
    stock = stock.upper()

    X_path = os.path.join(ARTIFACT_DIR, f"{stock}", f"{stock}_{interval}_X.npy")
    y_path = os.path.join(ARTIFACT_DIR, f"{stock}", f"{stock}_{interval}_y.npy")

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Training data not found. Run prepare_data.py first.")

    # LOAD DATA
    X = np.load(X_path)
    y = np.load(y_path)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # MODEL
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),

        LSTM(32),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse"
    )

    model.summary()

    # CALLBACKS
    model_path = os.path.join(
        MODEL_DIR, f"{stock}", f"{stock}_{interval}_lstm_model.h5"
    )

    callbacks = [
        EarlyStopping(
            monitor="loss",
            patience=5,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            model_path,
            monitor="loss",
            save_best_only=True
        )
    ]

    # TRAIN
    model.fit(
        X,
        y,
        epochs=50,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )

    print(f"Model trained and saved to {model_path}")


# ENTRY POINT
if __name__ == "__main__":
    stock = input("Enter stock ticker (e.g. ADANIENT): ").strip()
    interval = input("Enter interval (default: 1h): ").strip() or "1h"

    train_sequence_model(stock, interval)
