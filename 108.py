import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime
import os
import ccxt

# ==================================================
# CONFIG BITGET (SPOT)
# ==================================================
API_KEY = "bg_9ab333010434865ddb984599fcc7e6ed"
API_SECRET = "d25fdde596046266c05637f47d7e4764bd3ddf41f54d75d7e0d5d6027d42abdd"
API_PASSWORD = "nadi1968"

exchange = ccxt.bitget({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "password": API_PASSWORD,
    "enableRateLimit": True,
})

exchange.options["createMarketBuyOrderRequiresPrice"] = False

SYMBOL = "RIVER/USDT"
TIMEFRAME = "1m"
LIMIT = 500

# ==================================================
# MONTANT À TRADER
# ==================================================
USDT_PER_TRADE = 50

# ==================================================
# CONFIG TELEGRAM
# ==================================================
TELEGRAM_TOKEN = "8415781423:AAHScB-QqQWJHM3Re8IT62J9Y8HerMi_0Mo"
CHAT_ID = "5839604310"

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": msg}, timeout=5)
    except Exception as e:
        print("Telegram error:", e)

# ==================================================
# CSV TRADES
# ==================================================
CSV_FILE = "ema_biaisee_trades.csv"
if not os.path.exists(CSV_FILE):
    df_csv = pd.DataFrame(columns=["datetime", "type", "symbol", "qty", "price", "usdt_value", "reason"])
    df_csv.to_csv(CSV_FILE, index=False)

def save_trade_csv(trade_type, symbol, qty, price, reason):
    usdt_value = qty * price
    df_csv = pd.DataFrame([{
        "datetime": datetime.now(),
        "type": trade_type,
        "symbol": symbol,
        "qty": qty,
        "price": price,
        "usdt_value": usdt_value,
        "reason": reason
    }])
    df_csv.to_csv(CSV_FILE, mode='a', header=False, index=False)

# ==================================================
# ORDRES
# ==================================================
def buy_market(symbol, usdt_amount):
    try:
        ticker = exchange.fetch_ticker(symbol)
        buy_price = ticker['last']
        river_qty = usdt_amount / buy_price

        exchange.create_market_buy_order(symbol, usdt_amount)

        msg = f"✅ ACHAT\n{symbol}\nMontant: {usdt_amount} USDT\nPrix: {buy_price:.6f}\nQuantité: ~{river_qty:.6f} RIVER"
        send_telegram(msg)
        save_trade_csv("BUY", symbol, river_qty, buy_price, "Signal BUY")
        print(msg)
        return river_qty

    except Exception as e:
        msg = f"❌ ERREUR ACHAT: {e}"
        send_telegram(msg)
        print(msg)
        return 0

def sell_market(symbol, qty, reason="Signal SELL"):
    try:
        ticker = exchange.fetch_ticker(symbol)
        sell_price = ticker["last"]
        usdt_value = qty * sell_price

        exchange.create_market_sell_order(symbol, qty)

        msg = f"✅ VENTE\n{symbol}\nQuantité: {qty:.6f}\nPrix: {sell_price:.6f}\nMontant: ~{usdt_value:.2f} USDT"
        send_telegram(msg)
        save_trade_csv("SELL", symbol, qty, sell_price, reason)
        print(msg)
        return True

    except Exception as e:
        send_telegram(f"❌ ERREUR VENTE: {e}")
        print("ERREUR VENTE:", e)
        return False

# ==================================================
# STRATEGIE EMA vs BTC OVERLAY
# ==================================================
def strategy_ema_btc_overlay(df,
                              btc_timeframe="1m",
                              smooth_btc=True,
                              smooth_length=90,
                              norm_length=12,
                              use_filter=True,
                              filter_length=85,
                              ema_length=22,
                              use_sl=True,
                              sl_pct=3.5):
    df = df.copy()
    
    # === RÉCUPÉRATION BTC ===
    try:
        btc_ohlcv = exchange.fetch_ohlcv("BTC/USDT", btc_timeframe, limit=LIMIT)
        btc_df = pd.DataFrame(btc_ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
        btc_df["time"] = pd.to_datetime(btc_df["time"], unit="ms")
        
        df = df.merge(btc_df[["time", "close"]], on="time", how="left", suffixes=("", "_btc"))
        df["close_btc"] = df["close_btc"].ffill()
        
    except Exception as e:
        print(f"Erreur récupération BTC: {e}")
        df["close_btc"] = df["close"]
    
    min_btc = df["close_btc"].rolling(norm_length).min()
    max_btc = df["close_btc"].rolling(norm_length).max()
    min_asset = df["close"].rolling(norm_length).min()
    max_asset = df["close"].rolling(norm_length).max()
    
    df["btc_scaled"] = min_asset + (df["close_btc"] - min_btc) * (max_asset - min_asset) / (max_btc - min_btc)
    
    if smooth_btc:
        df["btc_smoothed"] = df["btc_scaled"].ewm(span=smooth_length, adjust=False).mean()
    else:
        df["btc_smoothed"] = df["btc_scaled"]
    
    df["btc_filter"] = df["btc_scaled"].ewm(span=filter_length, adjust=False).mean()
    df["ema_asset"] = df["close"].ewm(span=ema_length, adjust=False).mean()
    
    if use_filter:
        df["filter_ok"] = df["btc_smoothed"] > df["btc_filter"]
    else:
        df["filter_ok"] = True
    
    def crossover(a, b): return (a > b) & (a.shift(1) <= b.shift(1))
    def crossunder(a, b): return (a < b) & (a.shift(1) >= b.shift(1))
    
    df["entry_signal"] = crossover(df["ema_asset"], df["btc_smoothed"]) & df["filter_ok"]
    df["exit_signal"] = crossunder(df["ema_asset"], df["btc_smoothed"])
    
    position = 0
    entry_price = np.nan
    stop_price = np.nan
    buy_signal = []
    sell_signal = []
    
    for i in range(len(df)):
        buy, sell = False, False
        
        if df["exit_signal"].iloc[i] and position == 1:
            sell = True
            position = 0
        
        if df["entry_signal"].iloc[i] and position == 0:
            buy = True
            position = 1
            entry_price = df["close"].iloc[i]
            if use_sl:
                stop_price = entry_price * (1 - sl_pct / 100)
        
        if use_sl and position == 1 and df["low"].iloc[i] <= stop_price:
            sell = True
            position = 0
        
        buy_signal.append(buy)
        sell_signal.append(sell)
    
    df["buy"] = buy_signal
    df["sell"] = sell_signal
    
    return df

# ==================================================
# MAIN LOOP
# ==================================================
send_telegram("🚀 Bot EMA vs BTC Overlay lancé")

in_position = False
position_quantity = 0
last_candle_time = None

# ✅ CORRECTION DOUBLE VENTE : on ignore le solde RIVER existant
# Le bot repart toujours de zéro, il ne gère que ce qu'il achète lui-même
try:
    balance = exchange.fetch_balance()
    river_balance = balance["free"].get("RIVER", 0)
    if river_balance > 1:
        send_telegram(f"💼 RIVER détecté au démarrage: {river_balance:.6f}\nLe bot ignore ce solde et repart de zéro ✅")
        print(f"ℹ️ Solde RIVER ignoré: {river_balance:.6f} (le bot repart de zéro)")
except Exception as e:
    print(f"Erreur vérification solde: {e}")

while True:
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
        df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")

        df = df.iloc[:-1]

        if last_candle_time == df.iloc[-1]["time"]:
            time.sleep(1)
            continue

        last_candle_time = df.iloc[-1]["time"]

        df = strategy_ema_btc_overlay(df)
        row = df.iloc[-1]
        live_price = exchange.fetch_ticker(SYMBOL)["last"]

        # ACHAT
        if row["buy"] and not in_position:
            qty = buy_market(SYMBOL, USDT_PER_TRADE)
            if qty > 0:
                in_position = True
                position_quantity = qty

        # VENTE - seulement la quantité achetée par le bot
        elif row["sell"] and in_position:
            reason = "Exit EMA cross BTC" if row["exit_signal"] else "Stop Loss"
            if sell_market(SYMBOL, position_quantity * 0.997, reason):
                in_position = False
                position_quantity = 0

        print(f"{datetime.now()} | Close={row['close']:.6f} | Live={live_price:.6f} | EMA Asset={row['ema_asset']:.6f} | BTC Smooth={row['btc_smoothed']:.6f} | InPos={in_position}")

        time.sleep(1)

    except Exception as e:
        send_telegram(f"❌ ERREUR BOT\n{e}")
        print("ERREUR:", e)
        time.sleep(5)