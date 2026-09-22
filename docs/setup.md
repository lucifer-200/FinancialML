# Setup

This guide walks through setting up MarketPulse ML locally.

## Requirements

Use a recent Python version. Python 3.10 or newer is recommended.

The project uses common data science and ML packages, including:

- pandas
- numpy
- scikit-learn
- TensorFlow / Keras
- transformers
- torch
- yfinance
- matplotlib
- rich
- python-dotenv
- joblib
- pytz
- tqdm
- scipy

Install dependencies from the relevant module.

## Create A Virtual Environment

From the project root:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

## Install Live Engine Dependencies

```bash
cd models/live_engine
pip install -r requirements.txt
```

## Install Benchmark Dependencies

```bash
cd models/benchmark_lab
pip install -r requirements.txt
```

If you are using both modules in the same environment, install both requirement files.

## Configure API Keys

Copy `.env.example` into `.env` at the project root or inside the module you are running from:

```bash
copy .env.example .env
```

Fill in the keys you plan to use:

```text
GNEWS_API_KEY=
MARKETAUX_API_KEY=
NEWSDATA_API_KEY=
NEWSAPI_API_KEY=
```

The news loader checks these variables through `python-dotenv`.

## First Run

From the live engine folder:

```bash
python main.py
```

You should see the MarketPulse ML menu. Start with option `1` to run the model.

## Notes

The first run may take longer because transformer models can download weights locally. TensorFlow may also print hardware-related messages depending on your system.
