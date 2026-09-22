import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "benchmark_lab")
DATA_DIR = os.path.join(MODEL_DIR, "data")
output_file = os.path.join(DATA_DIR, "event_cleaned.csv")
input_file = os.path.join(DATA_DIR, "event_merged_signals.csv")

market_frame = pd.read_csv(input_file)

market_frame = market_frame[pd.to_numeric(market_frame["Close"], errors="coerce").notna()]

market_frame.to_csv(output_file, index=False)

print(f"Cleaned file saved to {output_file}")
