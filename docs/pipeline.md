# Pipeline

This document describes the live engine pipeline in the order it runs.

## 1. Price Data Loading

Script:

```text
models/live_engine/scripts/price_data_loader.py
```

This step downloads OHLCV candles for the selected stock and interval. It normalizes column names, converts timestamps, filters market hours, and saves a clean price file.

Output example:

```text
models/live_engine/data/TCS/TCS_1h_ohlcv.csv
```

## 2. News Loading

Script:

```text
models/live_engine/scripts/news_loader.py
```

This step fetches news using the configured providers and stock-specific keywords.

Output example:

```text
models/live_engine/data/TCS/TCS_news.csv
```

## 3. Event Filtering

Script:

```text
models/live_engine/scripts/event_filter.py
```

This step filters raw news into articles that are more likely to matter for the selected stock.

Output example:

```text
models/live_engine/data/TCS/TCS_news_filtered.csv
```

## 4. Signal Scoring

Script:

```text
models/live_engine/scripts/signal_score.py
```

This step uses a financial language model to score each filtered article.

Output example:

```text
models/live_engine/data/TCS/TCS_news_scored.csv
```

## 5. Feature Alignment

Script:

```text
models/live_engine/scripts/feature_alignment.py
```

This step aligns scored news with OHLCV candles and creates candle-level signal features.

Output example:

```text
models/live_engine/data/TCS/TCS_1h_lstm_input.csv
```

## 6. Feature Building

Script:

```text
models/live_engine/scripts/feature_builder.py
```

This step scales features, creates rolling windows, saves NumPy arrays, and writes scaler artifacts.

Output example:

```text
models/live_engine/artifacts/TCS/
```

## 7. Sequence Training

Script:

```text
models/live_engine/scripts/sequence_trainer.py
```

This step trains the LSTM model for the selected stock and interval.

Output example:

```text
models/live_engine/models/TCS/TCS_1h_lstm_model.h5
```

## 8. Forecast Inference

Script:

```text
models/live_engine/scripts/forecast_inference.py
```

This step loads the trained model and scalers, takes the latest feature window, predicts the next close, and returns a readable forecast summary.
