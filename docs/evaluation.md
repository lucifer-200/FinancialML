# Evaluation

The benchmark lab compares a baseline price-only model with a signal-enhanced model.

## Metrics

The evaluation script reports:

- MAE - mean absolute error.
- RMSE - root mean squared error.
- R2 - coefficient of determination.

These metrics are calculated in scaled target space.

## Benchmark Workflow

Run:

```bash
cd models/benchmark_lab
python scripts/forecast_evaluation.py
```

The script loads:

- `baseline_lstm.h5`
- `signal_lstm.h5`
- benchmark test arrays from `data/`

Then it prints both model scores and saves a comparison plot.

## Result Files

Result images are stored in:

```text
models/benchmark_lab/result/
```

Current result files:

- `prediction_overlay.png`
- `training_loss_profile.png`
- `baseline_vs_signal_forecast.png`

## Reading Results Carefully

A lower RMSE does not automatically mean the model is production-ready. It only means the model performed better on the current prepared test split.

Useful follow-up checks include:

- Test on more stocks.
- Test across different market periods.
- Compare stable periods with event-heavy periods.
- Track directional accuracy.
- Check whether the model is overfitting.

## Suggested Future Metrics

Good next metrics to add:

- Directional accuracy.
- Mean absolute percentage error.
- Hit rate on up/down moves.
- Error during high-news periods versus low-news periods.
- Confidence calibration.
