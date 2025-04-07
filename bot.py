import os
import time
import logging
import hmac
import hashlib
import requests
import numpy as np
import csv
from pathlib import Path
from urllib.parse import urlencode
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from rich.console import Console
from rich.table import Table
from rich.live import Live

# === CONFIG ===
BASE_URL = 'https://api.binance.com'
LOG_FILE = Path("trade_log.csv")
console = Console()

# === LOGGING ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === ENV VARS ===
API_KEY = os.getenv('BINANCE_API_KEY')
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')
SYMBOLS = os.getenv("TRADING_SYMBOLS", "BTCUSDT").upper().split(",")
INTERVAL_MINUTES = int(os.getenv("LOOP_INTERVAL_MINUTES", "5"))

# === UTILS ===
def send_email(subject, body):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_USER
    msg['To'] = EMAIL_USER
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        logging.info("Email sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

def get_server_time():
    return requests.get(BASE_URL + '/api/v3/time').json()['serverTime']

def sign_request(params):
    query_string = urlencode(params)
    signature = hmac.new(SECRET_KEY.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    return query_string + '&signature=' + signature

def get_klines(symbol, interval='1m', limit=100):
    url = f"{BASE_URL}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def moving_average_strategy(prices):
    short_window = 5
    long_window = 20
    if len(prices) < long_window:
        return 'HOLD'

    short_ma = np.mean(prices[-short_window:])
    long_ma = np.mean(prices[-long_window:])

    if short_ma > long_ma:
        return 'BUY'
    elif short_ma < long_ma:
        return 'SELL'
    else:
        return 'HOLD'

def log_trade(symbol, action, price, timestamp):
    header = ["timestamp", "symbol", "action", "price"]
    if not LOG_FILE.exists():
        with open(LOG_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, symbol, action, f"{price:.2f}"])

# === MAIN LOOP ===
def run_ai_trade_loop():
    with Live(console=console, refresh_per_second=1) as live:
        while True:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            table = Table(title="📈 AI Trading Bot Dashboard", style="bold white")
            table.add_column("Time", justify="center", style="cyan")
            table.add_column("Symbol", justify="center", style="green")
            table.add_column("Last Price", justify="right", style="yellow")
            table.add_column("Action", justify="center", style="magenta")

            for symbol in SYMBOLS:
                try:
                    klines = get_klines(symbol)
                    close_prices = [float(k[4]) for k in klines]
                    action = moving_average_strategy(close_prices)
                    latest_price = close_prices[-1]

                    action_display = {
                        "BUY": "[bold green]BUY[/]",
                        "SELL": "[bold red]SELL[/]",
                        "HOLD": "[bold white]HOLD[/]"
                    }.get(action, "[yellow]UNKNOWN[/]")

                    table.add_row(timestamp, symbol, f"{latest_price:.2f}", action_display)

                    if action != "HOLD":
                        send_email(
                            f"{symbol} Trade Signal",
                            f"Action: {action}\nPrice: {latest_price:.2f}\nTime: {timestamp}"
                        )
                        log_trade(symbol, action, latest_price, timestamp)

                except Exception as e:
                    logging.error(f"[{symbol}] Error: {e}")
                    table.add_row(timestamp, symbol, "[red]Error[/]", "[red]N/A[/]")

            live.update(table)
            time.sleep(INTERVAL_MINUTES * 60)

if __name__ == '__main__':
    run_ai_trade_loop()
