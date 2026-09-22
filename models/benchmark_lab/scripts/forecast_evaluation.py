import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "benchmark_lab")
DATA_DIR = os.path.join(MODEL_DIR, "data")
SAVE_DIR = os.path.join(MODEL_DIR, "saved_models")
RES_DIR = os.path.join(MODEL_DIR, "result")

baseline_model = load_model(os.path.join(SAVE_DIR,"baseline_lstm.h5"), compile=False)
signal_model = load_model(os.path.join(SAVE_DIR,"signal_lstm.h5"), compile=False)

Xn = np.load(os.path.join(DATA_DIR,"X_num_test.npy"))
yn = np.load(os.path.join(DATA_DIR,"y_num_test.npy"))
Xsig = np.load(os.path.join(DATA_DIR,"X_signal_test.npy"))
ysig = np.load(os.path.join(DATA_DIR,"y_signal_test.npy"))

baseline_predictions = baseline_model.predict(Xn).flatten()
signal_predictions = signal_model.predict(Xsig).flatten()

baseline_mae = mean_absolute_error(yn, baseline_predictions)
baseline_rmse = math.sqrt(mean_squared_error(yn, baseline_predictions))
baseline_r2 = r2_score(yn, baseline_predictions)

signal_mae = mean_absolute_error(ysig, signal_predictions)
signal_rmse = math.sqrt(mean_squared_error(ysig, signal_predictions))
signal_r2 = r2_score(ysig, signal_predictions)

print("\n===== MODEL PERFORMANCE (SCALED SPACE) =====")
print(f"Baseline model: MAE={baseline_mae:.4f}, RMSE={baseline_rmse:.4f}, R2={baseline_r2:.3f}")
print(f"Signal model: MAE={signal_mae:.4f}, RMSE={signal_rmse:.4f}, R2={signal_r2:.3f}")

if signal_rmse < baseline_rmse:
    print("Signal model outperforms baseline")
else:
    print("Baseline model is more stable")

plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.plot(yn,label="Actual"); plt.plot(baseline_predictions,label="Predicted")
plt.title("Baseline LSTM"); plt.legend()

plt.subplot(1,2,2)
plt.plot(ysig,label="Actual"); plt.plot(signal_predictions,label="Predicted")
plt.title("Signal-Enhanced LSTM"); plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(RES_DIR,"prediction_overlay.png"), dpi=300)
plt.show()
