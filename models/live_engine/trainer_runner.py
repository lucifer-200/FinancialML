import os
import sys
import subprocess

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.prompt import Prompt
from rich.rule import Rule

console = Console()


def header():
    console.print(
        Panel.fit(
            "[bold cyan]MarketPulse ML[/bold cyan]\n"
            "Market Signal Forecasting System\n\n"
            "[bold]Trainer Mode[/bold]",
            border_style="cyan"
        )
    )


def get_inputs():
    stock = Prompt.ask("Enter Stock Symbol").upper()
    interval = Prompt.ask("Enter Interval", default="1h")
    from_date = Prompt.ask("From Date (YYYY-MM-DD)")
    to_date = Prompt.ask("To Date (YYYY-MM-DD)")
    return stock, interval, from_date, to_date


def run_trainer_cmd(stock, interval, from_date, to_date):
    base_dir = os.path.dirname(__file__)
    trainer_path = os.path.join(base_dir, "trainer.py")

    python_exe = sys.executable

    cmd = [
        python_exe,
        trainer_path,
        stock,
        interval,
        from_date,
        to_date
    ]

    # Open new CMD window (Windows)
    subprocess.Popen(
        ["cmd.exe", "/k"] + cmd,
        cwd=base_dir
    )


def main():
    console.clear()
    header()

    stock, interval, from_date, to_date = get_inputs()

    console.print(Rule())

    phases = [
        "Stock Price Fetch",
        "News Fetch",
        "News Filtering",
        "Signal Scoring",
        "Feature Alignment",
        "Data Preparation",
        "LSTM Training",
    ]

    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console
    ) as progress:

        task = progress.add_task("Initializing Trainer", total=len(phases))

        for phase in phases:
            progress.update(task, description=phase)
            progress.advance(task)

    console.print("[green]Trainer launched in new terminal[/green]")
    console.print("[dim]You can continue using inference here.[/dim]\n")

    run_trainer_cmd(stock, interval, from_date, to_date)


if __name__ == "__main__":
    main()
