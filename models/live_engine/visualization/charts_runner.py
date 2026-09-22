import sys
from charts import show_charts

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python charts_runner.py STOCK INTERVAL")
        sys.exit(1)

    stock = sys.argv[1]
    interval = sys.argv[2]

    show_charts(stock, interval)
