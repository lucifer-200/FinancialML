import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "benchmark_lab")
input_file = os.path.join(MODEL_DIR, "data", "event_merged_raw.csv")
output_file = os.path.join(MODEL_DIR, "data", "event_merged_signals.csv")

market_frame = pd.read_csv(input_file)

MODEL = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

signals = []
scores = []

for text in tqdm(market_frame["Headline"].fillna("No news"), desc="Analyzing Sentiment"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = softmax(outputs.logits.detach().numpy()[0])
    
    labels = ['negative', 'neutral', 'positive']
    sentiment = labels[probs.argmax()]
    score = probs.max()
    
    signals.append(sentiment)
    scores.append(score)

market_frame["Signal_Label"] = signals
market_frame["Signal_Score"] = scores

market_frame.to_csv(output_file, index=False)
print(f"Sentiment analysis completed. Saved to {output_file}")
