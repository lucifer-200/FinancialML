# LOG SUPPRESSION (MUST BE FIRST)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import subprocess
from datetime import datetime, timedelta
import json

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.table import Table
from rich.prompt import Prompt
from rich.rule import Rule

from scripts.forecast_inference import run_forecast_inference

console = Console()

BASE_DIR = os.path.dirname(__file__)
SCRIPT_PATH = os.path.join(BASE_DIR, "scripts")
TRAINER_PATH = os.path.join(BASE_DIR, "trainer.py")
CHARTS_RUNNER = os.path.join(BASE_DIR, "visualization", "charts_runner.py")

# UI
def header(subtitle=""):
    console.print(
        Panel.fit(
            "[bold cyan]MarketPulse ML[/bold cyan]\n"
            "Market Signal Forecasting System"
            + (f"\n\n{subtitle}" if subtitle else ""),
            border_style="cyan"
        )
    )


def get_user_inputs():
    stock = Prompt.ask("Enter Stock Symbol").upper()
    interval = Prompt.ask("Enter Interval", default="1h")
    from_date = Prompt.ask("From Date (YYYY-MM-DD)")
    to_date = Prompt.ask("To Date (YYYY-MM-DD)")
    return stock, interval, from_date, to_date


# DATE UTILS
def compute_training_window(from_date: str, days: int = 7):
    start = datetime.strptime(from_date, "%Y-%m-%d")
    train_to = start - timedelta(days=1)
    train_from = train_to - timedelta(days=days - 1)
    return train_from.strftime("%Y-%m-%d"), train_to.strftime("%Y-%m-%d")


# TRAINER (SAME CMD)
def run_trainer_same_cmd(stock, interval, from_date):
    train_from, train_to = compute_training_window(from_date)

    phases = [
        "Fetching Stock Data",
        "Fetching News",
        "Filtering News",
        "Signal Scoring",
        "Feature Alignment",
        "Preparing LSTM Data",
        "Training LSTM"
    ]

    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console
    ) as progress:

        task = progress.add_task("Training...", total=len(phases))

        subprocess.run(
            [
                sys.executable,
                TRAINER_PATH,
                stock,
                interval,
                train_from,
                train_to
            ],
            check=True
        )

        progress.update(task, completed=len(phases))

    console.print("[green]Training completed successfully[/green]\n")


# RESULT
def show_result(result):
    table = Table(title="Result Summary", show_header=False)
    table.add_column("Metric")
    table.add_column("Value")

    for k, v in result.items():
        table.add_row(k, str(v))

    console.print(table)


# CHARTS
def launch_charts(stock, interval):
    subprocess.Popen(
        [sys.executable, CHARTS_RUNNER, stock, interval],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


# MENU
def menu():
    console.print(Rule())
    console.print(
        "[1] Run Model    "
        "[2] Show Charts    "
        "[3] Edit Stock Name and Keywords    "
        "[4] Edit News Keywords    "
        "[5] Exit"
    )
    return Prompt.ask("Select Option", choices=["1", "2", "3", "4", "5"])

# Editing Stock Keywords Menu
def stock_keywords_menu():
    console.print(Rule())
    console.print(
        "[1] Add Stock Keyword    "
        "[2] Remove Stock Keyword    "
        "[3] Remove Stock    "
        "[4] Add New Stock    "
        "[5] Back to Main Menu"
    )
    return Prompt.ask("Select Option", choices=["1", "2", "3", "4", "5"])

# Editing Event Keywords Menu
def event_keywords_menu():
    console.print(Rule())
    console.print(
        "[1] Add Event Keyword    "
        "[2] Remove Event Keyword    "
        "[3] Remove Event Type    "
        "[4] Add New Event Type    "
        "[5] Back to Main Menu"
    )
    return Prompt.ask("Select Option", choices=["1", "2", "3", "4", "5"])

# MAIN LOOP
def main():
    model_ran = False
    last_stock = None
    last_interval = None

    while True:
        header()

        choice = menu()

        # RUN MODEL
        if choice == "1":
            console.clear()
            header("Run Model")

            stock, interval, from_date, to_date = get_user_inputs()

            console.print(Rule())
            run_trainer_same_cmd(stock, interval, from_date)

            console.print(Rule())
            result = run_forecast_inference(stock, interval)
            show_result(result)

            model_ran = True
            last_stock = stock
            last_interval = interval

        # SHOW CHARTS
        elif choice == "2":
            if not model_ran:
                console.print("[red]Run the model first before viewing charts[/red]\n")
            else:
                launch_charts(last_stock, last_interval)

        # EDIT STOCK NAME AND KEYWORDS
        elif choice == "3":
            console.clear()
            header("Editor Mode - Stock Name and Keywords")

            try:
                with open(os.path.join(SCRIPT_PATH, "stock_keywords.json"), "r") as f:
                    data = json.load(f)

                while(True):
                    table = Table(title="Stock Keywords", show_header=True)
                    table.add_column("Stock Symbol", style="cyan", no_wrap=True)
                    table.add_column("Keywords", style="magenta")
                    for stock_symbol, keywords in data.items():
                        table.add_row(stock_symbol, ", ".join(keywords))
                    console.print(table)
                    selection = stock_keywords_menu()

                    # ADD STOCK KEYWORD
                    if selection == "1":
                        stock_name = Prompt.ask("Enter Stock Symbol to Add Keyword To").upper()
                        stock_symbols, keywords = data.keys(), data.values()

                        if stock_name not in stock_symbols:
                            console.print(f"[red]Stock Symbol {stock_name} not found.[/red]\n")
                            continue
                        else:
                            console.print(f"{stock_name} is available.\n")
                            console.print("Enter the keywords you want to add. To finish press ENTER on an empty line.")
                            while True:
                                keyword_to_add = Prompt.ask("Keyword to Add (or press ENTER to finish)").strip()
                                if keyword_to_add == "":
                                    break
                                elif keyword_to_add in data[stock_name]:
                                    console.print(f"[green]Keyword '{keyword_to_add}' already exists for {stock_name}.[/green]\n")
                                else:
                                    data[stock_name].append(keyword_to_add)
                                    with open(os.path.join(SCRIPT_PATH, "stock_keywords.json"), "w") as f:
                                        json.dump(data, f, indent=4)
                                    console.print(f"[green]Keyword '{keyword_to_add}' added to {stock_name}.[/green]\n")

                    # REMOVE STOCK KEYWORD
                    elif selection == "2":
                        stock_name = Prompt.ask("Enter Stock Symbol to Remove Keyword From").upper()
                        stock_symbols, keywords = data.keys(), data.values()

                        if stock_name not in stock_symbols:
                            console.print(f"[red]Stock Symbol {stock_name} not found.[/red]\n")
                            continue
                        else:
                            console.print(f"{stock_name} is available.\n")
                            console.print("Enter the keywords you want to remove. To finish press ENTER on an empty line.")
                            while True:
                                keyword_to_remove = Prompt.ask("Keyword to Remove (or press ENTER to finish)").strip()
                                if keyword_to_remove == "":
                                    break
                                elif keyword_to_remove not in data[stock_name]:
                                    console.print(f"[red]Keyword '{keyword_to_remove}' does not exist for {stock_name}.[/red]\n")
                                else:
                                    data[stock_name].remove(keyword_to_remove)
                                    with open(os.path.join(SCRIPT_PATH, "stock_keywords.json"), "w") as f:
                                        json.dump(data, f, indent=4)
                                    console.print(f"[green]Keyword '{keyword_to_remove}' removed from {stock_name}.[/green]\n")

                    # REMOVE STOCK
                    elif selection == "3":
                        stock_name = Prompt.ask("Enter Stock Symbol you want to remove").upper()
                        stock_symbols, keywords = data.keys(), data.values()

                        if stock_name not in stock_symbols:
                            console.print(f"[red]Stock Symbol {stock_name} not found.[/red]\n")
                            continue
                        else:
                            del data[stock_name]
                            with open(os.path.join(SCRIPT_PATH, "stock_keywords.json"), "w") as f:
                                json.dump(data, f, indent=4)
                            console.print(f"[green]Stock Symbol {stock_name} removed.[/green]\n")

                    # ADD NEW STOCK
                    elif selection == "4":
                        stock_name = Prompt.ask("Enter Stock Symbol you want to add").upper()
                        stock_symbols, keywords = data.keys(), data.values()

                        if stock_name in stock_symbols:
                            console.print(f"[red]Stock Symbol {stock_name} already exists.[/red]\n")
                            continue
                        else:
                            data[stock_name] = []
                            with open(os.path.join(SCRIPT_PATH, "stock_keywords.json"), "w") as f:
                                json.dump(data, f, indent=4)
                            console.print(f"[green]Stock Symbol {stock_name} added.[/green]\n")

                    # BACK TO MAIN MENU
                    elif selection == "5":
                        break

                    else:
                        console.print("[red]Invalid choice. Please try again.[/red]\n")

            except FileNotFoundError:
                console.print("[red]stock_keywords.json file not found.[/red]\n")
                continue

        # EDIT NEWS KEYWORDS
        elif choice == "4":
            console.clear()
            header("Editor Mode - NEWS Keywords")

            try:
                with open(os.path.join(SCRIPT_PATH, "event_keywords.json"), "r") as f:
                    event_data = json.load(f)

                while(True):
                    table = Table(title="Event Keywords", show_header=True)
                    table.add_column("Event Type", style="cyan", no_wrap=True)
                    table.add_column("Keywords", style="magenta")
                    for event_type, keywords in event_data.items():
                        table.add_row(event_type, ", ".join(keywords))
                    console.print(table)

                    selection = event_keywords_menu()

                    # ADD EVENT KEYWORD
                    if selection=="1":
                        event_type = Prompt.ask("Enter Event Type to Add Keyword To").upper()
                        event_types, keywords = event_data.keys(), event_data.values()

                        if event_type not in event_types:
                            console.print(f"[red]Event Type {event_type} not found.[/red]\n")
                            continue
                        else:
                            console.print(f"{event_type} is available.\n")
                            console.print("Enter the keywords you want to add. To finish press ENTER on an empty line.")
                            while True:
                                keyword_to_add = Prompt.ask("Keyword to Add (or press ENTER to finish)").strip()
                                if keyword_to_add == "":
                                    break
                                elif keyword_to_add in event_data[event_type]:
                                    console.print(f"[green]Keyword '{keyword_to_add}' already exists for {event_type}.[/green]\n")
                                else:
                                    event_data[event_type].append(keyword_to_add)
                                    with open(os.path.join(SCRIPT_PATH, "event_keywords.json"), "w") as f:
                                        json.dump(event_data, f, indent=4)
                                    console.print(f"[green]Keyword '{keyword_to_add}' added to {event_type}.[/green]\n")

                    # REMOVE EVENT KEYWORD
                    elif selection=="2":
                        event_type = Prompt.ask("Enter Event Type to Remove Keyword From").upper()
                        event_types, keywords = event_data.keys(), event_data.values()

                        if event_type not in event_types:
                            console.print(f"[red]Event Type {event_type} not found.[/red]\n")
                            continue
                        else:
                            console.print(f"{event_type} is available.\n")
                            console.print("Enter the keywords you want to remove. To finish press ENTER on an empty line.")
                            while True:
                                keyword_to_remove = Prompt.ask("Keyword to Remove (or press ENTER to finish)").strip()
                                if keyword_to_remove == "":
                                    break
                                elif keyword_to_remove not in event_data[event_type]:
                                    console.print(f"[red]Keyword '{keyword_to_remove}' does not exist for {event_type}.[/red]\n")
                                else:
                                    event_data[event_type].remove(keyword_to_remove)
                                    with open(os.path.join(SCRIPT_PATH, "event_keywords.json"), "w") as f:
                                        json.dump(event_data, f, indent=4)
                                    console.print(f"[green]Keyword '{keyword_to_remove}' removed from {event_type}.[/green]\n")

                    # REMOVE EVENT TYPE
                    elif selection=="3":
                        event_type = Prompt.ask("Enter Event Type you want to remove").upper()
                        event_types, keywords = event_data.keys(), event_data.values()

                        if event_type not in event_types:
                            console.print(f"[red]Event Type {event_type} not found.[/red]\n")
                            continue
                        else:
                            del event_data[event_type]
                            with open(os.path.join(SCRIPT_PATH, "event_keywords.json"), "w") as f:
                                json.dump(event_data, f, indent=4)
                            console.print(f"[green]Event Type {event_type} removed.[/green]\n")

                    # ADD NEW EVENT TYPE
                    elif selection=="4":
                        event_type = Prompt.ask("Enter Event Type you want to add").upper()
                        event_types, keywords = event_data.keys(), event_data.values()

                        if event_type in event_types:
                            console.print(f"[red]Event Type {event_type} already exists.[/red]\n")
                            continue
                        else:
                            event_data[event_type] = []
                            with open(os.path.join(SCRIPT_PATH, "event_keywords.json"), "w") as f:
                                json.dump(event_data, f, indent=4)
                            console.print(f"[green]Event Type {event_type} added.[/green]\n")

                    # BACK TO MAIN MENU
                    elif selection=="5":
                        break

                    else:
                        console.print("[red]Invalid choice. Please try again.[/red]\n")

            except FileNotFoundError:
                console.print("[red]event_keywords.json file not found.[/red]\n")
                continue

        # EXIT
        elif choice == "5":
            console.print("\n[bold cyan]Exiting MarketPulse ML[/bold cyan]")
            sys.exit(0)

        # INVALID CHOICE
        else:
            console.print("[red]Invalid choice. Please try again.[/red]\n")


# ENTRY
if __name__ == "__main__":
    main()
