import streamlit as st
import logging
from plotly.subplots import make_subplots
logging.basicConfig(level=logging.INFO)
logging.info("Checking if plotly.subplots is imported...")
try:
    import plotly.subplots
    logging.info("plotly.subplots is imported")
except ImportError:
    logging.error("plotly.subplots is NOT imported")
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import pytz
import ta
from utils.telegram_alerts import send_telegram_message, create_ssl_cci_ema_signal_info
from utils.charting import create_ssl_cci_ema_chart, create_rsi_ema_chart
from utils.indicator_utils import convert_to_utc_plus_2, calculate_cci, get_current_time
from utils.data_fetcher import get_data
from indicators.ema_cci_ssl import calculate_ssl_cci_ema_signals
from indicators.rsi_ema import calculate_rsi_ema_signals

# Set page config
st.set_page_config(
    page_title="EMA CCI SSL BUY SELL Signal [THANHCONG]",
    layout="wide"
)

# App title and description
st.title("EMA CCI SSL BUY SELL Signal [THANHCONG]")
st.markdown("Python version of TradingView PineScript indicator")

# Fixed symbol and timeframe as per requirements
symbol = "GC=F"
interval = "5m"

# Send initialization message

# Auto-refresh functionality
if 'last_refresh_time' not in st.session_state:
    st.session_state.last_refresh_time = datetime.now()

# Store previous signals to avoid duplicate alerts
if 'last_buy_signal_time' not in st.session_state:
    st.session_state.last_buy_signal_time = None
if 'last_sell_signal_time' not in st.session_state:
    st.session_state.last_sell_signal_time = None

# Create a placeholder for the last refresh time
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = get_current_time()

# Auto-refresh functionality - check if 5 minutes have passed
current_time = datetime.now()
if (current_time - st.session_state.last_refresh_time).total_seconds() >= 300:  # 300 seconds = 5 minutes
    st.session_state.last_refresh_time = current_time
    st.session_state.last_refresh = get_current_time()
    st.rerun()

# Get data
data = get_data(symbol, interval, "7d")  # Get 7 days of 5-minute data

# Check if data is empty
if data.empty:
    st.error(f"No data available for {symbol}")
    st.stop()

# Calculate indicators for SSL/CCI/EMA
df_ssl_cci_ema = calculate_ssl_cci_ema_signals(data.copy())

# Calculate indicators for RSI/EMA
df_rsi_ema = calculate_rsi_ema_signals(data.copy())

# Display the SSL/CCI/EMA chart
st.subheader(f"Chart: {symbol} (5m) - UTC+2 Timezone")
chart_ssl_cci_ema = create_ssl_cci_ema_chart(df_ssl_cci_ema, symbol)
st.plotly_chart(chart_ssl_cci_ema, use_container_width=True)

# Display SSL/CCI/EMA signal info
st.subheader("SSL/CCI/EMA Signal Information")
signal_info_ssl_cci_ema, is_profit_ssl_cci_ema = create_ssl_cci_ema_signal_info(df_ssl_cci_ema, symbol)

if signal_info_ssl_cci_ema is not None:
    # Style the DataFrame
    def color_signal_type(val):
        color = 'green' if val == 'Buy' else 'red'
        return f'background-color: {color}; color: white'

    def color_change(val):
        color = 'green' if is_profit_ssl_cci_ema else 'red'
        return f'background-color: {color}; color: white'

    # Apply styling - using newer Pandas styling API
    styled_signal_info_ssl_cci_ema = signal_info_ssl_cci_ema.style.map(color_signal_type, subset=['Signal Type'])
    styled_signal_info_ssl_cci_ema = styled_signal_info_ssl_cci_ema.map(color_change, subset=['Change %'])

    st.dataframe(styled_signal_info_ssl_cci_ema, use_container_width=True)
else:
    st.info("No SSL/CCI/EMA signals detected in the current data.")

# Add RSI/EMA chart
st.subheader("RSI and RSI EMA")

# Create RSI chart
chart_rsi_ema = create_rsi_ema_chart(df_rsi_ema)
st.plotly_chart(chart_rsi_ema, use_container_width=True)

# Display last updated time
st.write(f"Last updated: {st.session_state.last_refresh}")

# Auto-refresh every 5 minutes (using Streamlit's rerun mechanism)
time.sleep(500)  # Small delay to prevent excessive CPU usage
st.rerun()
