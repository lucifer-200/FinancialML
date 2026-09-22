# Usage

MarketPulse ML can be used in two ways: through the live CLI or through the benchmark scripts.

## Live Engine

The live engine is the easiest entry point.

```bash
cd models/live_engine
python main.py
```

The CLI offers:

- Run Model
- Show Charts
- Edit Stock Name and Keywords
- Edit News Keywords
- Exit

### Running A Forecast

Choose `Run Model` and enter:

- Stock symbol, for example `TCS`
- Interval, for example `1h`
- From date, for example `2026-01-12`
- To date, for example `2026-01-17`

The engine computes a short training window before the forecast range, fetches price and news data, scores signals, builds LSTM input, trains the model, and runs inference.

The output includes:

- Last close
- Predicted close
- Expected range
- Expected change
- Trend
- Volatility risk
- Confidence
- News count
- Average signal

### Showing Charts

After running the model, choose `Show Charts`. This opens price and signal charts for the last run stock and interval.

## Benchmark Lab

The benchmark lab is script-driven. Run the scripts from `models/benchmark_lab`.

Typical order:

```bash
cd models/benchmark_lab
python scripts/market_data_download.py
python scripts/event_data_merge.py
python scripts/event_signal_scoring.py
python scripts/event_data_cleaning.py
python scripts/sequence_dataset_builder.py
python scripts/sequence_model_training.py
python scripts/forecast_evaluation.py
```

The benchmark produces:

- Trained baseline and signal models.
- Sequence datasets.
- Evaluation metrics.
- Result plots in `models/benchmark_lab/result`.

## Running Individual Live Steps

The live engine is modular. You can call individual scripts if needed:

```bash
python scripts/price_data_loader.py
python scripts/news_loader.py
python scripts/event_filter.py
python scripts/signal_score.py
python scripts/feature_alignment.py
python scripts/feature_builder.py
python scripts/sequence_trainer.py
python scripts/forecast_inference.py
```

For normal use, `main.py` is preferred because it runs the stages in the correct order.
