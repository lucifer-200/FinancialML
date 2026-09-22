# Data

MarketPulse ML uses two kinds of data: market data and news/event data.

## Live Engine Data

Live engine data is stored under:

```text
models/live_engine/data/<STOCK>/
```

Example:

```text
models/live_engine/data/TCS/
```

### OHLCV Files

Files ending in `_ohlcv.csv` contain market candles.

Common columns:

- `timestamp`
- `timestamp_ist`
- `open`
- `high`
- `low`
- `close`
- `volume`

### News Files

Files ending in `_news.csv` contain raw normalized news.

Common columns:

- `Date`
- `Time`
- `Headline`
- `Description`
- `Source`

### Filtered News

Files ending in `_news_filtered.csv` contain articles that passed keyword and event filtering.

Additional columns may include:

- `RelevanceScore`
- `EventType`

### Scored News

Files ending in `_news_scored.csv` contain scored event records.

Important columns:

- `SignalScore`
- `SignalLabel`

### LSTM Input

Files ending in `_lstm_input.csv` contain aligned market and signal features.

Important columns:

- `open`
- `high`
- `low`
- `close`
- `volume`
- `signal_mean`
- `signal_min`
- `signal_max`
- `signal_std`
- `news_count`
- `neg_count`

## Benchmark Lab Data

Benchmark data is stored under:

```text
models/benchmark_lab/data/
```

Important files:

- `event_price_window.csv` - event-window price data.
- `event_news.csv` - event-window news records.
- `event_merged_raw.csv` - merged price and news data before scoring.
- `event_merged_signals.csv` - merged data with signal labels and scores.
- `event_cleaned.csv` - cleaned model-ready event data.
- `X_num_train.npy`, `X_num_test.npy` - baseline model inputs.
- `X_signal_train.npy`, `X_signal_test.npy` - signal-enhanced model inputs.
- `y_num_train.npy`, `y_num_test.npy` - baseline targets.
- `y_signal_train.npy`, `y_signal_test.npy` - signal model targets.

## Data Hygiene

Do not commit private API keys, proprietary datasets, or files that contain account-level information.

Generated model artifacts can become large. If this project grows, consider keeping trained model files outside Git or using a model registry.
