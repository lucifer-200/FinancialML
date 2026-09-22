import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import random

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "benchmark_lab")
DATA_DIR = os.path.join(MODEL_DIR, "data")
SAVE_DIR = os.path.join(MODEL_DIR, "saved_models")
RES_DIR = os.path.join(MODEL_DIR, "result")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

Xn_tr = np.load(os.path.join(DATA_DIR,"X_num_train.npy"))
Xn_te = np.load(os.path.join(DATA_DIR,"X_num_test.npy"))
yn_tr = np.load(os.path.join(DATA_DIR,"y_num_train.npy"))
yn_te = np.load(os.path.join(DATA_DIR,"y_num_test.npy"))

Xsig_tr = np.load(os.path.join(DATA_DIR,"X_signal_train.npy"))
Xsig_te = np.load(os.path.join(DATA_DIR,"X_signal_test.npy"))
ysig_tr = np.load(os.path.join(DATA_DIR,"y_signal_train.npy"))
ysig_te = np.load(os.path.join(DATA_DIR,"y_signal_test.npy"))

def build_model(shape, lr):
    m = Sequential([
        LSTM(64, return_sequences=True, input_shape=shape),
        Dropout(0.25),
        LSTM(32),
        Dense(1)
    ])
    m.compile(optimizer=Adam(lr), loss="mse", metrics=["mae"])
    return m

early = EarlyStopping(patience=10, restore_best_weights=True)

print("Training baseline LSTM")
baseline_model = build_model((Xn_tr.shape[1], Xn_tr.shape[2]), 0.001)
baseline_history = baseline_model.fit(Xn_tr, yn_tr, validation_data=(Xn_te, yn_te),
                                      epochs=100, batch_size=16, callbacks=[early])

baseline_model.save(os.path.join(SAVE_DIR,"baseline_lstm.h5"))

print("Training signal-enhanced LSTM")
signal_model = build_model((Xsig_tr.shape[1], Xsig_tr.shape[2]), 0.0007)
signal_history = signal_model.fit(Xsig_tr, ysig_tr, validation_data=(Xsig_te, ysig_te),
                                  epochs=120, batch_size=16, callbacks=[early])

signal_model.save(os.path.join(SAVE_DIR,"signal_lstm.h5"))

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(baseline_history.history["loss"]); plt.plot(baseline_history.history["val_loss"])
plt.title("Baseline LSTM")

plt.subplot(1,2,2)
plt.plot(signal_history.history["loss"]); plt.plot(signal_history.history["val_loss"])
plt.title("Signal-Enhanced LSTM")

plt.tight_layout()
plt.savefig(os.path.join(RES_DIR,"training_loss_profile.png"), dpi=300)
plt.show()
