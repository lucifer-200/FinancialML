# Architecture

MarketPulse ML is split into two main workspaces: a live forecasting engine and a benchmark lab.

The live engine is built for end-to-end operation. It starts with a stock symbol and date range, collects market and news data, converts news into numerical signal features, trains a sequence model, and returns a forecast summary.

The benchmark lab is more controlled. It is meant for comparing two modeling approaches on a prepared event dataset:

- A baseline model trained on price and volume features.
- A signal-enhanced model trained on price, volume, and market signal features.

## High-Level Flow

```text
Stock Symbol + Date Range
        |
        v
Price Data Loader
        |
        v
News Loader
        |
        v
Event Filter
        |
        v
Signal Scoring
        |
        v
Feature Alignment
        |
        v
Sequence Dataset Builder
        |
        v
LSTM Training
        |
        v
Forecast Inference
```

## Main Components

### Price Data

Price data is loaded as OHLCV candles. The live engine currently works with files shaped around:

- `timestamp`
- `timestamp_ist`
- `open`
- `high`
- `low`
- `close`
- `volume`

### News Data

News is collected from configured providers, normalized into a shared CSV format, and stored per stock symbol.

The raw news files include:

- `Date`
- `Time`
- `Headline`
- `Description`
- `Source`

### Event Filtering

Not every article is useful. The filtering step uses stock-specific keywords and event keywords to keep news that is more likely to matter for the target symbol.

### Signal Scoring

Filtered news is passed through a financial language model and converted into:

- `SignalScore`
- `SignalLabel`

The score is later aggregated into candle-level features.

### Feature Alignment

News and market candles live on different timelines. The alignment step maps news items into the relevant candle window and creates features such as:

- `signal_mean`
- `signal_min`
- `signal_max`
- `signal_std`
- `news_count`
- `neg_count`

### Sequence Modeling

The project uses LSTM models because the input is sequential. Each model sees a rolling window of recent candles and predicts the next close value in scaled space.

## Design Notes

The project intentionally keeps the pipeline modular. Each stage can be replaced without rewriting the full system. For example:

- `yfinance` can be replaced with another market data provider.
- The signal model can be swapped for a different financial NLP model.
- The LSTM can be replaced by a Transformer, GRU, or tree-based sequence approach.
- The CLI can later be wrapped with an API or dashboard.
