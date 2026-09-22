import os
import sys
import warnings

# ------------------ SILENCE TF LOGS ------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, BASE_DIR)

# ------------------ IMPORT PIPELINE STEPS ------------------
from scripts.price_data_loader import download_stock_data
from scripts.news_loader import fetch_all_news
from scripts.event_filter import filter_news_for_stock
from scripts.signal_score import score_news_for_stock
from scripts.feature_alignment import align_news_with_ohlcv
from scripts.feature_builder import build_lstm_dataset
from scripts.sequence_trainer import train_sequence_model


def train_stock_pipeline(
    stock: str,
    interval: str,
    from_date: str,
    to_date: str,
):
    stock = stock.upper()

    # Phase 1: Stock OHLCV
    download_stock_data(
        symbol=stock,
        start_date=from_date,
        end_date=to_date,
        interval=interval,
    )
    print("__PHASE_DONE__:FETCH_STOCK", flush=True)

    # Phase 2: News Fetching
    fetch_all_news(
        stock=stock,
        from_date=from_date,
        to_date=to_date,
    )
    print("__PHASE_DONE__:FETCH_NEWS", flush=True)

    # Phase 3: News Filtering
    filter_news_for_stock(stock)
    print("__PHASE_DONE__:FILTER_NEWS", flush=True)

    # Phase 4: Signal Scoring
    score_news_for_stock(stock)
    print("__PHASE_DONE__:SIGNAL", flush=True)

    # Phase 5: Alignment
    align_news_with_ohlcv(stock, interval)
    print("__PHASE_DONE__:ALIGNMENT", flush=True)

    # Phase 6: LSTM Dataset
    build_lstm_dataset(stock, interval)
    print("__PHASE_DONE__:PREP", flush=True)

    # Phase 7: Training
    train_sequence_model(stock, interval)
    print("__PHASE_DONE__:TRAIN", flush=True)


# CLI ENTRY
if __name__ == "__main__":
    stock = sys.argv[1]
    interval = sys.argv[2]
    from_date = sys.argv[3]
    to_date = sys.argv[4]

    train_stock_pipeline(stock, interval, from_date, to_date)
