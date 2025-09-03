import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Helper functions ---
def sma(series, window): 
    return series.rolling(window).mean()

def ema(series, span): 
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def bollinger(series, window=20, num_std=2):
    ma = sma(series, window)
    std = series.rolling(window).std(ddof=0)
    return ma, ma + num_std*std, ma - num_std*std

# --- Streamlit UI ---
st.set_page_config(page_title="Stock Analyst", layout="wide")
st.title("📈 Stock Analyst — Streamlit App")

with st.sidebar:
    tickers_input = st.text_input("Tickers (comma separated)", "AAPL, MSFT")
    start = st.date_input("Start date", pd.to_datetime("2020-01-01"))
    end = st.date_input("End date", pd.to_datetime("today"))
    run = st.button("Analyze")

if run:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    st.write(f"Downloading: {', '.join(tickers)} from {start} to {end} ...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # Pick Adjusted Close if available
    if isinstance(data, dict) or ("Adj Close" in data.columns):
        prices = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
    else:
        prices = data["Close"] if "Close" in data.columns else data

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    for t in prices.columns:
        st.subheader(f"{t}")
        px = prices[t].dropna()
        if px.empty:
            st.write("No data for this ticker/time range.")
            continue

        df = pd.DataFrame({"Price": px})
        df["SMA20"] = sma(px, 20)
        df["SMA50"] = sma(px, 50)
        df["RSI14"] = rsi(px)
        bb_ma, bb_up, bb_lo = bollinger(px)
        df["BB_MA"] = bb_ma
        df["BB_UP"] = bb_up
        df["BB_LO"] = bb_lo

        # Price + indicators plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df["Price"], label="Price")
        ax.plot(df.index, df["SMA20"], label="SMA20")
        ax.plot(df.index, df["SMA50"], label="SMA50")
        ax.plot(df.index, df["BB_UP"], "--", label="BB Up")
        ax.plot(df.index, df["BB_LO"], "--", label="BB Lo")
        ax.legend(loc="upper left")
        ax.set_title(f"{t} Price & Indicators")
        st.pyplot(fig)

        # RSI plot
        fig2, ax2 = plt.subplots(figsize=(10, 2.5))
        ax2.plot(df.index, df["RSI14"], label="RSI(14)")
        ax2.axhline(70, linestyle="--")
        ax2.axhline(30, linestyle="--")
        ax2.set_title("RSI(14)")
        st.pyplot(fig2)

        st.dataframe(df.tail(10))
      
