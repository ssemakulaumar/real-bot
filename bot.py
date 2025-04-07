import os
import requests
import hmac
import hashlib
import time
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from urllib.parse import urlencode
import numpy as np

# Load environment variables
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
BINANCE_BASE_URL = "https://api.binance.com"

EMAIL_HOST = "smtp.gmail.com"
EMAIL_PORT = 587
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", EMAIL_USER)

# Logging
logging.basicConfig(
    filename="trade_decisions.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

TRADING_STRATEGY = {
    "trade_quantity": 0.01,
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30
}

def send_email_notification(subject, message):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))

        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()

        logging.info(f"Email sent: {subject}")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

def binance_request(endpoint, method='GET', params=None, signed=False):
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}

    if params is None:
        params = {}

    if signed:
        params['timestamp'] = int(time.time() * 1000)
        query_string = urlencode(params)
        signature = hmac.new(BINANCE_SECRET_KEY.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        params['signature'] = signature

    url = f"{BINANCE_BASE_URL}{endpoint}?{urlencode(params)}"
    response = requests.request(method, url, headers=headers)
    return response.json()

def fetch_market_price(symbol):
    endpoint = "/api/v3/ticker/price"
    params = {"symbol": symbol}
    response = binance_request(endpoint, params=params)
    return float(response.get("price", 0))

def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = 100 - 100 / (1 + rs)
    return rsi

def ai_trade_decision(symbol):
    prices = [fetch_market_price(symbol) for _ in range(TRADING_STRATEGY['rsi_period'] + 1)]
    rsi = calculate_rsi(prices, TRADING_STRATEGY["rsi_period"])

    if rsi < TRADING_STRATEGY['rsi_oversold']:
        return "BUY"
    elif rsi > TRADING_STRATEGY['rsi_overbought']:
        return "SELL"
    return None

def place_trade(symbol, side, quantity):
    endpoint = "/api/v3/order"
    params = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": quantity
    }
    response = binance_request(endpoint, method='POST', params=params, signed=True)
    return response

def run_ai_trade():
    symbol = os.getenv("TRADING_SYMBOL", "BTCUSDT").upper()
    action = ai_trade_decision(symbol)
    quantity = TRADING_STRATEGY["trade_quantity"]

    if action is None:
        print("No trade action taken based on AI analysis.")
        return

    try:
        response = place_trade(symbol, action, quantity)
        logging.info(f"AI Trade executed: {response}")
        send_email_notification("Trade Executed", f"Action: {action}\nSymbol: {symbol}\nQuantity: {quantity}\nResponse: {response}")
        print("Trade executed successfully.")
        print(response)
    except Exception as e:
        logging.error(f"Trade execution failed: {str(e)}")
        send_email_notification("Trade Execution Failed", f"Error: {str(e)}")
        print("Trade execution failed.")
        print(str(e))

if __name__ == '__main__':
    run_ai_trade()
