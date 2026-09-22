import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# PATH SETUP
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "live_engine", "data")

# MODEL SETUP
MODEL_NAME = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.eval()

LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


# SIGNAL FUNCTION
def compute_signal(text):
    """
    Returns:
    - signal_score (float)
    - signal_label (str)
    """

    if not text or not isinstance(text, str):
        return 0.0, "neutral"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    neg, neu, pos = probs.tolist()

    # signal score in [-1, 1]
    signal_score = pos - neg

    signal_label = LABEL_MAP[torch.argmax(probs).item()]

    return signal_score, signal_label


# MAIN PIPELINE
def score_news_for_stock(stock):
    stock = stock.upper()

    input_file = os.path.join(DATA_DIR, f"{stock}", f"{stock}_news_filtered.csv")
    output_file = os.path.join(DATA_DIR, f"{stock}", f"{stock}_news_scored.csv")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)

    if df.empty:
        print("No news to score.")
        return

    scores = []
    labels = []

    print(f"Scoring market signals for {len(df)} news articles...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = f"{row['Headline']} {row['Description']}"
        score, label = compute_signal(text)
        scores.append(score)
        labels.append(label)

    df["SignalScore"] = scores
    df["SignalLabel"] = labels

    df.to_csv(output_file, index=False)

    print(f"Signal-scored news saved to {output_file}")


# ENTRY POINT
if __name__ == "__main__":
    stock = input("Enter stock ticker (e.g. ADANIENT): ").strip()
    score_news_for_stock(stock)
