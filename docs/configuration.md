# Configuration

MarketPulse ML keeps most project-specific settings in JSON files and environment variables.

## Environment Variables

News API keys are read from environment variables:

```text
GNEWS_API_KEY
MARKETAUX_API_KEY
NEWSDATA_API_KEY
NEWSAPI_API_KEY
```

Use `.env.example` as a starting point.

## Stock Keywords

File:

```text
models/live_engine/scripts/stock_keywords.json
```

This file maps stock symbols to keywords used during news collection.

Example structure:

```json
{
  "TCS": [
    "TCS",
    "Tata Consultancy Services"
  ]
}
```

Add more aliases when a company is commonly referenced in multiple ways.

## Event Keywords

File:

```text
models/live_engine/scripts/event_keywords.json
```

This file defines event categories and the words that should trigger them.

Typical categories might include:

- financial results
- governance
- acquisitions
- regulation
- lawsuits
- leadership changes

## CLI Editing

The live engine includes menu options for editing stock keywords and event keywords directly from the CLI.

Run:

```bash
cd models/live_engine
python main.py
```

Then use:

- `Edit Stock Name and Keywords`
- `Edit News Keywords`

## Model Settings

Some model parameters are currently defined directly in scripts:

- Window size in `feature_builder.py`.
- LSTM architecture in `sequence_trainer.py`.
- Benchmark sequence length in `sequence_dataset_builder.py`.

If the project grows, these can be moved into a central config file.
