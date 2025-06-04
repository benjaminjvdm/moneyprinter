import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import yfinance as yf
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import ta
from datetime import timedelta

# Utility functions for indicator calculations

def convert_to_utc_plus_2(dt):
    if isinstance(dt, pd.Timestamp):
        # Check if the datetime already has timezone info
        if dt.tzinfo is not None:
            # Already has timezone, convert to UTC+2
            return dt.astimezone(pytz.timezone('Etc/GMT-2'))
        else:
            # Naive datetime, localize to UTC first
            utc_time = pytz.utc.localize(dt)
            return utc_time.astimezone(pytz.timezone('Etc/GMT-2'))
    return dt

def calculate_cci(df, n):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = tp.rolling(window=n).mean()
    md = tp.rolling(window=n).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    cci = (tp - ma) / (0.015 * md)
    return cci

# Add auto-refresh functionality
def get_current_time():
    # Get current time in UTC
    utc_now = datetime.now(pytz.utc)
    # Convert to UTC+2
    utc_plus_2 = utc_now.astimezone(pytz.timezone('Etc/GMT-2'))
    return utc_plus_2.strftime('%Y-%m-%d %H:%M:%S')

# Utility function for fetching data
@st.cache_data(ttl=300)  # Cache data for 5 minutes
def get_data(ticker, interval, period):
    data = yf.download(ticker, interval=interval, period=period)
    # Flatten multi-index columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    # Convert index to UTC+2
    data = data.reset_index()
    # Check if 'Date' column exists (for daily data) and rename to 'Datetime'
    if 'Date' in data.columns:
        data.rename(columns={'Date': 'Datetime'}, inplace=True)
    
    # Ensure 'Datetime' column exists before applying conversion
    if 'Datetime' in data.columns:
        data['Datetime'] = data['Datetime'].apply(convert_to_utc_plus_2)
    else:
        st.error("Error: 'Datetime' column not found in fetched data.")
        return pd.DataFrame() # Return empty DataFrame to prevent further errors
    return data

# Utility functions for creating charts

# Create Plotly figure for SSL/CCI/EMA
def create_ssl_cci_ema_chart(df, symbol):
    # Create candlestick chart
    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['Datetime'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=symbol
        )
    )

    # Add SSL Channel
    fig.add_trace(
        go.Scatter(
            x=df['Datetime'],
            y=df['sslUp'],
            name='SSL Up',
            line=dict(color='green', width=2)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df['Datetime'],
            y=df['sslDown'],
            name='SSL Down',
            line=dict(color='red', width=2)
        )
    )

    # Add regression channels
    # Filter out NaN values
    reg_df = df.dropna(subset=['reg_mid', 'reg_upper', 'reg_lower'])

    if not reg_df.empty:
        # Add regression mid line
        fig.add_trace(
            go.Scatter(
                x=reg_df['Datetime'],
                y=reg_df['reg_mid'],
                name='Regression Mid',
                line=dict(color='blue', width=2, dash='solid')
            )
        )

        # Add regression upper line
        fig.add_trace(
            go.Scatter(
                x=reg_df['Datetime'],
                y=reg_df['reg_upper'],
                name='Regression Upper',
                line=dict(color='blue', width=1, dash='dash')
            )
        )

        # Add regression lower line
        fig.add_trace(
            go.Scatter(
                x=reg_df['Datetime'],
                y=reg_df['reg_lower'],
                name='Regression Lower',
                line=dict(color='blue', width=1, dash='dash')
            )
        )

    # Add buy signals
    buy_signals = df[df['buy_signal']]
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals['Datetime'],
                y=buy_signals['Low'] * 0.999,  # Slightly below the low
                mode='markers+text',
                marker=dict(symbol='triangle-up', size=15, color='green'),
                text="Buy",
                textposition="bottom center",
                name='Buy Signal'
            )
        )

    # Add sell signals
    sell_signals = df[df['sell_signal']]
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals['Datetime'],
                y=sell_signals['High'] * 1.001,  # Slightly above the high
                mode='markers+text',
                marker=dict(symbol='triangle-down', size=15, color='red'),
                text="Sell",
                textposition="top center",
                name='Sell Signal'
            )
        )

    # Update layout
    fig.update_layout(
        title=f'EMA CCI SSL BUY SELL Signal [THANHCONG] - {symbol} (5m) - UTC+2 Timezone',
        xaxis_title='Date (UTC+2)',
        yaxis_title='Price',
        height=800,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Format x-axis to show time in UTC+2
    fig.update_xaxes(
        tickformat="%Y-%m-%d %H:%M:%S",
        tickangle=-45
    )

    return fig

# Logic for EMA CCI SSL BUY SELL Signal indicator
def calculate_ssl_cci_ema_signals(df):
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # === SSL Channel from HTF ===
    # For 5m timeframe, HTF is 60m (1h) according to the PineScript
    htf_data = df.copy()
    # Resample to 1h timeframe
    htf_data = htf_data.set_index('Datetime')
    htf_data = htf_data.resample('1h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    # Calculate SMA for high and low
    htf_data['smaHigh'] = htf_data['High'].rolling(window=8).mean()
    htf_data['smaLow'] = htf_data['Low'].rolling(window=8).mean()

    # Calculate Hlv
    htf_data['Hlv'] = 0.0
    for i in range(len(htf_data)):
        if i == 0:
            htf_data.iloc[i, htf_data.columns.get_loc('Hlv')] = np.nan
        else:
            close = htf_data.iloc[i]['Close']
            smaHigh = htf_data.iloc[i]['smaHigh']
            smaLow = htf_data.iloc[i]['smaLow']
            prev_hlv = htf_data.iloc[i-1]['Hlv']

            if close > smaHigh:
                htf_data.iloc[i, htf_data.columns.get_loc('Hlv')] = 1
            elif close < smaLow:
                htf_data.iloc[i, htf_data.columns.get_loc('Hlv')] = -1
            else:
                htf_data.iloc[i, htf_data.columns.get_loc('Hlv')] = prev_hlv

    # Calculate SSL Up and Down
    htf_data['sslDown'] = np.where(htf_data['Hlv'] < 0, htf_data['smaHigh'], htf_data['smaLow'])
    htf_data['sslUp'] = np.where(htf_data['Hlv'] < 0, htf_data['smaLow'], htf_data['smaHigh'])

    # Merge HTF data back to original dataframe
    htf_data = htf_data.reset_index()
    htf_data = htf_data[['Datetime', 'sslDown', 'sslUp']]

    # Forward fill HTF data to match 5m timeframe
    df = df.set_index('Datetime')
    htf_data = htf_data.set_index('Datetime')

    # Merge and forward fill
    df = pd.merge_asof(df.reset_index(), htf_data.reset_index(), on='Datetime', direction='backward')
    df = df.set_index('Datetime')
    df[['sslDown', 'sslUp']] = df[['sslDown', 'sslUp']].ffill()  # Using ffill instead of fillna(method='ffill')

    # === RSI Filter ===
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Calculate EMA of RSI (from second app)
    df['RSI_EMA'] = pd.Series(df['RSI']).ewm(span=14, adjust=False).mean()

    rsiOverbought = 70
    rsiOversold = 30
    df['rsi_filter_buy'] = df['RSI'] < rsiOversold
    df['rsi_filter_sell'] = df['RSI'] > rsiOverbought

    # === CCI Settings ===
    # Use calculate_cci from utils
    df['cciTurbo'] = calculate_cci(df, 6)  # CCI Turbo Length = 6
    df['cci14'] = calculate_cci(df, 14)    # CCI 14 Length = 14

    # === Volume Spike Filter ===
    df['vol_ma'] = df['Volume'].rolling(window=20).mean()
    df['vol_spike'] = df['Volume'] > df['vol_ma'] * 1.5

    # === CCI Divergence (Simplified) ===
    df['bullish_div'] = (df['Close'] < df['Close'].shift(1)) & (df['cciTurbo'] > df['cciTurbo'].shift(1))
    df['bearish_div'] = (df['Close'] > df['Close'].shift(1)) & (df['cciTurbo'] < df['cciTurbo'].shift(1))

    # === Candlestick Reversal Patterns ===
    # Hammer (bottom): small body, long lower wick
    df['hammer'] = (df['Close'] > df['Open']) & ((df['Open'] - df['Low']) > 2 * (df['Close'] - df['Open']))

    # Shooting Star (top): small body, long upper wick
    df['shooting_star'] = (df['Open'] > df['Close']) & ((df['High'] - df['Open']) > 2 * (df['Open'] - df['Close']))

    # === Final Combined Conditions ===
    df['enhanced_buy_signal'] = df['rsi_filter_buy'] & df['vol_spike'] & df['bullish_div'] & df['hammer']
    df['enhanced_sell_signal'] = df['rsi_filter_sell'] & df['vol_spike'] & df['bearish_div'] & df['shooting_star']

    # Slope calculation
    df['slope_60'] = df['Close'] - df['Close'].shift(60)

    # === Entry Conditions ===
    df['buy_condition'] = (df['slope_60'] < -2) & df['enhanced_buy_signal'] & (df['sslUp'] > df['sslDown']) & (df['Close'] > df['sslUp']) & (df['Close'].shift(1) < df['sslUp'].shift(1))
    df['sell_condition'] = (df['slope_60'] > 2) & df['enhanced_sell_signal'] & (df['sslUp'] < df['sslDown']) & (df['Close'] < df['sslUp']) & (df['Close'].shift(1) > df['sslUp'].shift(1))

    # === Cross detection for entry signals ===
    df['ssl_up_cross_down'] = (df['sslUp'] > df['sslDown']) & (df['sslUp'].shift(1) <= df['sslDown'].shift(1))
    df['ssl_down_cross_up'] = (df['sslUp'] < df['sslDown']) & (df['sslUp'].shift(1) >= df['sslDown'].shift(1))

    # First time only logic - using infer_objects() to avoid FutureWarning
    df['was_ssl_up_cross_down'] = df['ssl_up_cross_down'].shift(1).fillna(False)
    df['was_ssl_down_cross_up'] = df['ssl_down_cross_up'].shift(1).fillna(False)

    df['buy_signal'] = df['ssl_up_cross_down'] & ~df['was_ssl_up_cross_down']
    df['sell_signal'] = df['ssl_down_cross_up'] & ~df['was_ssl_down_cross_up']

    # === Regression Channel ===
    # Initialize columns for regression channel
    df['reg_mid'] = np.nan
    df['reg_upper'] = np.nan
    df['reg_lower'] = np.nan
    df['reg_slope'] = np.nan

    # We need at least 60 candles for the regression channel
    length = 60
    devlen = 2.0

    if len(df) >= length:
        for i in range(length, len(df)):
            # Extract the window of close prices directly as a numpy array
            window_data = df['Close'].iloc[i-length:i].values
            x = np.arange(length)

            # Calculate linear regression
            try:
                slope, intercept = np.polyfit(x, window_data, 1)

                # Calculate the regression line
                reg_line = intercept + slope * x

                # Calculate the standard deviation of residuals
                dev = np.sqrt(np.sum((window_data - reg_line)**2) / length)

                # Calculate the end point (current candle)
                midline = reg_line[-1]  # Last point of regression line
                upper_line = midline + devlen * dev
                lower_line = midline - devlen * dev

                # Store values
                df.iloc[i, df.columns.get_loc('reg_mid')] = midline
                df.iloc[i, df.columns.get_loc('reg_upper')] = upper_line
                df.iloc[i, df.columns.get_loc('reg_lower')] = lower_line
                df.iloc[i, df.columns.get_loc('reg_slope')] = slope
            except:
                # Skip this window if there's an error
                continue

    return df.reset_index()

import logging
from plotly.subplots import make_subplots
logging.basicConfig(level=logging.INFO)
logging.info("Checking if plotly.subplots is imported...")
try:
    import plotly.subplots
    logging.info("plotly.subplots is imported")
except ImportError:
    logging.error("plotly.subplots is NOT imported")

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
interval = "5m" # This interval is still used for the 4h chart data fetching

# Store previous signals to avoid duplicate alerts
if 'last_buy_signal_time' not in st.session_state:
    st.session_state.last_buy_signal_time = None
if 'last_sell_signal_time' not in st.session_state:
    st.session_state.last_sell_signal_time = None

# Manual refresh button
if st.button("Refresh Charts"):
    st.session_state.last_refresh = get_current_time()
    st.rerun()

# Initialize last_refresh if not present (for initial load)
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = get_current_time()

# Create two columns for side-by-side charts
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Chart: {symbol} (4h) - UTC+2 Timezone")
    data_4h_raw = get_data(symbol, "1h", "60d") # Fetch 1-hour data for 60 days
    if data_4h_raw.empty:
        st.error(f"No data available for {symbol} (4h)")
    else:
        # Resample to 4-hour data
        data_4h = data_4h_raw.set_index('Datetime').resample('4h').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna().reset_index()
        
        if data_4h.empty:
            st.error(f"No 4-hour aggregated data available for {symbol}")
        else:
            df_ssl_cci_ema_4h = calculate_ssl_cci_ema_signals(data_4h.copy())
            chart_ssl_cci_ema_4h = create_ssl_cci_ema_chart(df_ssl_cci_ema_4h, symbol)
            st.plotly_chart(chart_ssl_cci_ema_4h, use_container_width=True)

with col2:
    st.subheader(f"Chart: {symbol} (1d) - UTC+2 Timezone")
    data_1d = get_data(symbol, "1d", "1y") # Use "1d" interval for daily data, fetch 1 year
    if data_1d.empty:
        st.error(f"No data available for {symbol} (1d)")
    else:
        df_ssl_cci_ema_1d = calculate_ssl_cci_ema_signals(data_1d.copy())
        chart_ssl_cci_ema_1d = create_ssl_cci_ema_chart(df_ssl_cci_ema_1d, symbol)
        st.plotly_chart(chart_ssl_cci_ema_1d, use_container_width=True)

# Display last updated time
st.write(f"Last updated: {st.session_state.last_refresh}")