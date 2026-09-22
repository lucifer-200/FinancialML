# Troubleshooting

This page covers common issues when running MarketPulse ML.

## API Key Errors

If news loading fails, check your `.env` file.

Make sure the keys are named exactly:

```text
GNEWS_API_KEY
MARKETAUX_API_KEY
NEWSDATA_API_KEY
NEWSAPI_API_KEY
```

Also check whether the provider account has quota remaining.

## No News Found

This can happen when:

- The date range is too narrow.
- The stock keywords are too strict.
- API quota is exhausted.
- The selected provider has no matching articles.

Try adding aliases in `stock_keywords.json`.

## Missing Model File

If inference says a model file is missing, run training first.

From the live engine:

```bash
python main.py
```

Choose `Run Model`. The training step should create model files under:

```text
models/live_engine/models/<STOCK>/
```

## Not Enough Data

The LSTM needs enough rows to build rolling windows. If the selected date range is too short, feature building or inference can fail.

Use a longer training period or a smaller interval if available.

## TensorFlow Messages

TensorFlow may print CPU/GPU or oneDNN messages. The project suppresses most logs, but some environment-specific messages can still appear.

These messages are usually harmless unless the script stops with an exception.

## Transformer Model Download Is Slow

The first signal scoring run may download model weights. After that, the files are usually cached locally.

## Charts Do Not Open

Charts depend on your local Python and Matplotlib backend. If charts do not open, try running the chart script directly:

```bash
cd models/live_engine
python visualization/charts_runner.py TCS 1h
```

## CSV Columns Missing

If a script complains about missing columns, the data may have been created by an older version of the pipeline.

Regenerate the pipeline from the start:

1. Price data loading.
2. News loading.
3. Event filtering.
4. Signal scoring.
5. Feature alignment.
6. Feature building.

## Clean Restart

For a clean live-engine run, remove generated files for a specific stock and rerun the CLI. Keep this scoped to the target stock folder so you do not delete unrelated data.
