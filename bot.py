import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import time
import logging
import hmac
import hashlib 
import numpy as np
import csv
import requests
from pathlib import Path
from urllib.parse import urlencode
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from rich.console import Console
from rich.table import Table
from rich.live import Live
import logging

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
SYMBOLS = os.getenv("TRADING_SYMBOLS", "EURUSDT").upper().split(",")
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
        logging.error("Failed to send email", exc_info=True)

def get_server_time():
    return requests.get(BASE_URL + '/api/v3/time').json()['serverTime']

def sign_request(params):
    query_string = urlencode(params)
    signature = hmac.new(SECRET_KEY.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    return query_string + '&signature=' + signature

def get_klines_df(symbol, interval='1m', limit=1000):
    """Fetches kline data from Binance and returns a pandas DataFrame."""
    url = "{}/api/v3/klines?symbol={}&interval={}&limit={}".format(BASE_URL, symbol, interval, limit)

    try:
        response = requests.get(url)
        response.raise_for_status()
        raw = response.json()
        df = pd.DataFrame(raw, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except Exception as e:
        logging.error("Error fetching kline data for {}: {}".format(symbol, e))
        return pd.DataFrame()

def get_current_price(symbol):
    try:
        url = f"{BASE_URL}/api/v3/ticker/price?symbol={symbol}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return float(data["price"])
    except Exception as e:
        logging.error(f"Failed to get price for {symbol}: {e}")
        return None

def get_asset_balance(asset):
    try:
        timestamp = get_server_time()
        params = {
            "timestamp": timestamp
        }
        signed_query = sign_request(params)
        headers = {"X-MBX-APIKEY": API_KEY}
        url = f"{BASE_URL}/api/v3/account?{signed_query}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        balances = response.json()["balances"]
        for b in balances:
            if b["asset"] == asset:
                return float(b["free"])
        return 0.0
    except Exception as e:
        logging.error(f"Failed to get balance for {asset}: {e}")
        return 0.0

def log_trade(symbol, action, price, timestamp):
    try:
        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, symbol, action, price])
        logging.info(f"Logged trade: {action} {symbol} at {price}")
    except Exception as e:
        logging.error(f"Failed to log trade: {e}")


def feature_engineering(data):
    """Generate features for the model."""
    try:
        data['SMA_10'] = data['close'].rolling(window=10).mean()
        data['SMA_50'] = data['close'].rolling(window=50).mean()
        data['RSI'] = calculate_rsi(data['close'])
        data['target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)
        data = data.dropna()
        return data
    except Exception as e:
        logging.error(f"Error during feature engineering: {e}")
        return pd.DataFrame()

def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index (RSI)."""
    try:
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    except Exception as e:
        logging.error(f"Error calculating RSI: {e}")
        return pd.Series()

def train_model(data):
    """Train the AI model."""
    try:
        features = ['SMA_10', 'SMA_50', 'RSI']
        X = data[features]
        y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logging.info(f"Model Accuracy: {model.score(X_test, y_test) * 100:.2f}%")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        return None

def execute_trade(signal, symbol, quantity):
    """Execute trades based on signal."""
    try:
        if signal == "BUY":
            broker.place_order(symbol=symbol, quantity=quantity, side="buy")
            logging.info(f"BUY order placed for {quantity} of {symbol}")
        elif signal == "SELL":
            broker.place_order(symbol=symbol, quantity=quantity, side="sell")
            logging.info(f"SELL order placed for {quantity} of {symbol}")
    except Exception as e:
        logging.error(f"Error executing trade: {e}")

def auto_rebalance(symbol, target_base_ratio=0.5):
    """Rebalance the portfolio to maintain a target base/quote ratio."""
    base_asset, quote_asset = split_symbol(symbol)
    base_balance = get_asset_balance(base_asset)
    quote_balance = get_asset_balance(quote_asset)

    # Get the current price to calculate how much base/quote we can buy/sell
    current_price = get_current_price(symbol)
    if current_price is None:
        return

    # Current ratio of base asset
    total_value = base_balance * current_price + quote_balance
    current_base_ratio = (base_balance * current_price) / total_value

    # If the ratio is off from the target, rebalance
    if current_base_ratio > target_base_ratio:
        # Too much base asset, sell base asset for quote asset
        amount_to_sell = base_balance - (target_base_ratio * total_value / current_price)
        execute_trade("SELL", symbol, amount_to_sell)
    elif current_base_ratio < target_base_ratio:
        # Too much quote asset, buy base asset with quote asset
        amount_to_buy = (target_base_ratio * total_value - base_balance * current_price) / current_price
        execute_trade("BUY", symbol, amount_to_buy)

def momentum_strategy(prices, threshold=0.01):
    """Momentum Strategy: Buy if price is increasing, sell if decreasing."""
    if len(prices) < 2:
        return 'HOLD'

    momentum = (prices[-1] - prices[-2]) / prices[-2]

    if momentum > threshold:
        return 'BUY'
    elif momentum < -threshold:
        return 'SELL'
    else:
        return 'HOLD'

def breakout_strategy(prices, window=20):
    """Breakout Strategy: Buy if price breaks resistance, sell if below support."""
    if len(prices) < window:
        return 'HOLD'

    highest_high = max(prices[-window:])
    lowest_low = min(prices[-window:])
    
    if prices[-1] > highest_high:
        return 'BUY'
    elif prices[-1] < lowest_low:
        return 'SELL'
    else:
        return 'HOLD'

def split_symbol(symbol):
    """Split the symbol into base and quote assets."""
    base = symbol[:-3]
    quote = symbol[-3:]
    return base, quote

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
            table.add_column("Quote Bal", justify="right", style="blue")
            table.add_column("Base Bal", justify="right", style="bright_blue")

            for symbol in SYMBOLS:
                try:
                    df = get_klines_df(symbol)
                    if df.empty:
                        raise ValueError("No kline data returned.")

                    close_prices = df['close'].tolist()

                    
                    # Choose Strategy
                    action = momentum_strategy(close_prices)  
                    # Or breakout_strategy(close_prices)
                    
                    latest_price = close_prices[-1]

                    base_asset, quote_asset = split_symbol(symbol)
                    base_balance = get_asset_balance(base_asset)
                    quote_balance = get_asset_balance(quote_asset)

                    action_display = {
                        "BUY": "[bold green]BUY[/]",
                        "SELL": "[bold red]SELL[/]",
                        "HOLD": "[bold white]HOLD[/]"
                    }.get(action, "[yellow]UNKNOWN[/]")

                    table.add_row(
                        timestamp,
                        symbol,
                        f"{latest_price:.4f}",
                        action_display,
                        f"{quote_balance:.2f}",
                        f"{base_balance:.4f}"
                    )

                    # Auto-Rebalance for portfolio
                    auto_rebalance(symbol, target_base_ratio=0.5)

                    if action != "HOLD":
                        send_email(
                            f"{symbol} Trade Signal",
                            f"Action: {action}\nPrice: {latest_price:.4f}\nTime: {timestamp}"
                        )
                        log_trade(symbol, action, latest_price, timestamp)

                except Exception as e:
                    logging.error(f"[{symbol}] Error: {e}")
                    table.add_row(timestamp, symbol, "[red]Error[/]", "[red]N/A[/]", "N/A", "N/A")

            live.update(table)
            time.sleep(INTERVAL_MINUTES * 60)

if __name__ == '__main__':
    run_ai_trade_loop()
