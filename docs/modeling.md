# Modeling

MarketPulse ML uses LSTM models to learn from short sequences of market candles.

The modeling approach is intentionally simple. The goal is to keep the pipeline understandable while still allowing price and event context to work together.

## Baseline Model

The baseline model uses only numerical market features:

- Open
- High
- Low
- Close
- Volume

This model answers a basic question: how well can recent price behavior explain the next close?

## Signal-Enhanced Model

The signal-enhanced model extends the baseline feature set with event-driven signal features.

In the live engine, those features include:

- `signal_mean`
- `signal_min`
- `signal_max`
- `signal_std`
- `news_count`
- `neg_count`

In the benchmark lab, the event dataset includes a `Signal_Score` column that is scaled and appended to the numerical features.

## LSTM Architecture

The benchmark model uses a compact stacked LSTM:

```text
Input sequence
    |
    v
LSTM(64, return_sequences=True)
    |
    v
Dropout
    |
    v
LSTM(32)
    |
    v
Dense(1)
```

The model predicts the next scaled close price.

## Scaling

The pipeline uses `MinMaxScaler` for model inputs and targets.

The live engine stores scalers under:

```text
models/live_engine/artifacts/<STOCK>/
```

The inference step loads the same scalers to convert model output back into a price estimate.

## Windowing

The live engine uses a fixed rolling window defined in `feature_builder.py`.

The benchmark lab uses its own sequence length in `sequence_dataset_builder.py`.

These values are easy to tune, but changing them means the model should be retrained.

## Practical Limits

The models are lightweight and useful for experimentation. They are not designed to handle every market regime. Forecast quality depends heavily on:

- Data volume.
- News coverage.
- Event relevance.
- Time interval.
- Market volatility.
- Whether the selected stock has enough useful history.
