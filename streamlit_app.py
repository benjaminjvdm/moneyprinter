import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import ta

# Set Streamlit app title
st.title("GBPJPY Candlestick Chart with RSI and EMA")

# Function to fetch data
@st.cache_data
def get_data():
    data = yf.download(tickers="GBPJPY=X", period="7d", interval="5m")
    return data

# Get the data
data = get_data()

# Calculate RSI
rsi = ta.momentum.RSIIndicator(data['Close'].squeeze(), window=14).rsi()

# Calculate EMA of RSI
rsi_ema = pd.Series(rsi).ewm(span=14, adjust=False).mean()

# Create subplots
fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1)

# Add RSI trace
fig.add_trace(go.Scatter(x=data.index[-24:], y=rsi[-24:], name="RSI", line=dict(color="lightblue")), row=1, col=1)

# Add EMA of RSI trace
fig.add_trace(go.Scatter(x=data.index[-24:], y=rsi_ema[-24:], name="RSI EMA", line=dict(color="hotpink")), row=1, col=1)

# Find crossover points
rsi_series = pd.Series(rsi)
rsi_ema_series = pd.Series(rsi_ema)
crossovers = []
for i in range(1, len(rsi_series[-24:])):
    if (rsi_series[-24:][i] > rsi_ema_series[-24:][i] and rsi_series[-24:][i-1] < rsi_ema_series[-24:][i-1]):
        if rsi_series[-24:][i] > 50:
            crossovers.append({'index': data.index[-24:][i], 'type': 'bullish', 'rsi': rsi_series[-24:][i]})
    elif (rsi_series[-24:][i] < rsi_ema_series[-24:][i] and rsi_series[-24:][i-1] > rsi_ema_series[-24:][i-1]):
        if rsi_series[-24:][i] < 50:
            crossovers.append({'index': data.index[-24:][i], 'type': 'bearish', 'rsi': rsi_series[-24:][i]})

# Add crossover markers
for crossover in crossovers:
    if crossover['type'] == 'bullish':
        fig.add_trace(go.Scatter(x=[crossover['index']], y=[crossover['rsi'] - 5], mode='markers',
                                 marker=dict(symbol='triangle-up', size=10, color='green'),
                                 name='Bullish Crossover'), row=1, col=1)
    elif crossover['type'] == 'bearish':
        fig.add_trace(go.Scatter(x=[crossover['index']], y=[crossover['rsi'] + 5], mode='markers',
                                 marker=dict(symbol='triangle-down', size=10, color='red'),
                                 name='Bearish Crossover'), row=1, col=1)

# Update layout
fig.update_layout(
    xaxis_rangeslider_visible=False,
    yaxis=dict(range=[0, 100]),
    shapes = [dict(
        x0=data.index[-24:][0], x1=data.index[-24:][-1], y0=50, y1=50, type="line",
        line=dict(color="white", width=2, dash="dash")
    )]
)

# Show the plot
st.plotly_chart(fig, use_container_width=True)

# Display last updated time
utc_time = time.gmtime()
hour = (utc_time.tm_hour + 2) % 24
day = utc_time.tm_mday
if utc_time.tm_hour + 2 >= 24:
    day += 1
last_update_time = time.strftime("%Y-%m-%d " + str(hour).zfill(2) + ":%M:%S", utc_time)
st.write(f"Last updated: {last_update_time}")

# Auto-refresh every 5 minutes
time.sleep(300)
st.rerun()