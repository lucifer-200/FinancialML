# MarketPulse ML

MarketPulse ML is a Python-based market forecasting project that combines price action, financial news, event filtering, market signal scoring, and LSTM sequence models.

The project is built around a simple idea: price history is useful, but it often misses the context behind sudden market movement. News, filings, corporate events, policy shifts, and broader market narratives can all change how a stock behaves. MarketPulse ML turns that external context into structured signal features and feeds them into forecasting models alongside OHLCV market data.

This repository contains two working parts:

- `models/live_engine` - an interactive pipeline for loading market data, collecting news, building features, training a model, and running a forecast.
- `models/benchmark_lab` - a controlled benchmark workspace for comparing a baseline LSTM model against a signal-enhanced LSTM model.

The code is intended for experimentation, learning, and research-style analysis. It is not financial advice and should not be used as the only basis for trading or investment decisions.

## What It Does

MarketPulse ML can:

- Download stock price data.
- Fetch recent financial news from configured news APIs.
- Filter news using stock and event keywords.
- Score news into market signal values.
- Align those signals with OHLCV candles.
- Build LSTM-ready sequence datasets.
- Train forecasting models.
- Run inference for the next expected close.
- Produce basic charts and benchmark plots.

## Repository Layout

```text
MarketPulse ML/
|
├── models/
│   ├── live_engine/
│   │   ├── main.py
│   │   ├── trainer.py
│   │   ├── scripts/
│   │   ├── visualization/
│   │   └── data/
│   |
│   └── benchmark_lab/
│       ├── scripts/
│       ├── data/
│       └── result/
|
├── docs/
│   ├── architecture.md
│   ├── setup.md
│   ├── usage.md
│   ├── data.md
│   ├── modeling.md
│   ├── pipeline.md
│   ├── configuration.md
│   ├── evaluation.md
│   └── troubleshooting.md
|
├── .env.example
└── README.md
```

## Quick Start

Create a virtual environment and install the dependencies for the engine you want to run.

```bash
cd models/live_engine
pip install -r requirements.txt
python main.py
```

For the benchmark workspace:

```bash
cd models/benchmark_lab
pip install -r requirements.txt
```

Then run the scripts in `models/benchmark_lab/scripts` in the order described in [docs/usage.md](docs/usage.md).

## Environment Variables

The live news pipeline can use multiple news providers. Create a `.env` file from `.env.example` and add the keys you have:

```text
GNEWS_API_KEY=
MARKETAUX_API_KEY=
NEWSDATA_API_KEY=
NEWSAPI_API_KEY=
```

The pipeline can still be edited and tested locally without all providers, but live news collection depends on valid API keys.

## Documentation

- [Architecture](docs/architecture.md)
- [Setup](docs/setup.md)
- [Usage](docs/usage.md)
- [Data](docs/data.md)
- [Modeling](docs/modeling.md)
- [Pipeline](docs/pipeline.md)
- [Configuration](docs/configuration.md)
- [Evaluation](docs/evaluation.md)
- [Troubleshooting](docs/troubleshooting.md)

## Disclaimer

This project is for educational and experimental use. Forecasts are estimates produced by statistical and machine learning models. They can be wrong, especially in fast-moving or low-data market conditions.
