import os
import time
import logging
import hmac
import hashlib
import requests
import numpy as np
from urllib.parse import urlencode
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
API_KEY = os.getenv('BINANCE_API_KEY')
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')

BASE_URL = 'https://api.binance.com'

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
    return response.json()

def moving_average_strategy(prices):
    short_window = 5
    long_window = 20

    short_ma = np.mean(prices[-short_window:])
    long_ma = np.mean(prices[-long_window:])

    if short_ma > long_ma:
        return 'BUY'
    elif short_ma < long_ma:
        return 'SELL'
    else:
        return 'HOLD'

def run_ai_trade():
    symbol = os.getenv("TRADING_SYMBOL", "BTCUSDT").upper()
    logging.info(f"Running strategy for {symbol}...")

    try:
        klines = get_klines(symbol)
        close_prices = [float(k[4]) for k in klines]
        action = moving_average_strategy(close_prices)

        logging.info(f"Strategy Decision: {action}")
        send_email(f"{symbol} Trade Signal", f"Action: {action}\nLatest Price: {close_prices[-1]}")
    except Exception as e:
        logging.error(f"Error running trade strategy: {e}")
        send_email("AI Bot Error", str(e))

def run_ai_trade_loop():
    try:
        interval = int(os.getenv("LOOP_INTERVAL_MINUTES", "5"))
    except ValueError:
        interval = 5

    while True:
        logging.info("Starting new trade cycle...")
        run_ai_trade()
        logging.info(f"Sleeping for {interval} minutes...")
        time.sleep(interval * 60)

if __name__ == '__main__':
    run_ai_trade_loop()
