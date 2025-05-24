import asyncio
import telegram
# Function to send Telegram message
def send_telegram_message(message):
    print("Inside send_telegram_message with message:", message)  # Log statement
    # WARNING: Hardcoding the token is not recommended. Use st.secrets instead.
    # Ensure TELEGRAM_BOT_TOKEN is set in Streamlit's secrets management.
    bot = telegram.Bot(token=st.secrets["TELEGRAM_BOT_TOKEN"])
    try:
        result = asyncio.run(bot.send_message(chat_id="@MilitechKD637", text=message))
        st.write("Telegram message sent successfully!")
        print("Telegram message sent successfully!")  # Log statement
        print("Telegram send_message result:", result)  # Log statement
        print("Telegram bot object:", bot)  # Log statement
    except Exception as e:
        st.write(f"Error sending Telegram message: {e}")
        print(f"Error sending Telegram message: {e}")  # Log statement
    print("Exiting send_telegram_message")  # Log statement

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import ta

# Load the DataFrame from CSV file
try:
    message_log_df = pd.read_csv("message_log.csv")
except FileNotFoundError:
    message_log_df = pd.DataFrame(columns=['message'])

def normalize_message(message):
    # Remove leading/trailing whitespace and convert to lowercase
    return message.strip().lower()

def is_duplicate_message(message, message_log_df):
    normalized_message = normalize_message(message)
    return normalized_message in message_log_df['message'].values

def add_message_to_log(message, message_log_df):
    normalized_message = normalize_message(message)
    new_row = pd.DataFrame([{'message': normalized_message}])
    message_log_df = pd.concat([message_log_df, new_row], ignore_index=True)

    # Limit the DataFrame to the last 10 messages
    if len(message_log_df) > 10:
        message_log_df = message_log_df.iloc[-10:].reset_index(drop=True)

    message_log_df.to_csv("message_log.csv", index=False)
    return message_log_df

# Set Streamlit app title
st.title("GBPJPY Candlestick Chart with RSI and EMA")

# Function to fetch data
@st.cache_data
def get_data():
    data = yf.download(tickers="BTC-USD", period="7d", interval="5m")
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
fig.add_trace(go.Scatter(x=data.index[-48:], y=rsi[-48:], name="RSI", line=dict(color="lightblue")), row=1, col=1)

# Add EMA of RSI trace
fig.add_trace(go.Scatter(x=data.index[-48:], y=rsi_ema[-48:], name="RSI EMA", line=dict(color="hotpink")), row=1, col=1)

# Find crossover points
rsi_series = pd.Series(rsi)
rsi_ema_series = pd.Series(rsi_ema)
crossovers = []
for i in range(1, len(rsi_series[-48:])):
    if (rsi_series[-48:][i] > rsi_ema_series[-48:][i] and rsi_series[-48:][i-1] < rsi_ema_series[-48:][i-1]):
        if rsi_series[-48:][i] > 50:
            # Check if RSI was below 50 before the crossover
            if any(rsi_series[-48:][max(0, i-7):i] < 50):
                crossovers.append({'index': data.index[-48:][i], 'type': 'bullish', 'rsi': rsi_series[-48:][i]})

            # Send Telegram message on bullish crossover
            message = f"Bullish Crossover Alert!\nDate/Time: {data.index[-48:][i]}\nRSI: {rsi_series[-48:][i]:.2f}"
            if not is_duplicate_message(message, message_log_df):
                print("Calling send_telegram_message with message:", message)  # Log statement
                send_telegram_message(message)
                message_log_df = add_message_to_log(message, message_log_df)
            else:
                print("Duplicate message suppressed:", message)  # Log statement
    elif (rsi_series[-48:][i] < rsi_ema_series[-48:][i] and rsi_series[-48:][i-1] > rsi_ema_series[-48:][i-1]):
        if rsi_series[-48:][i] < 50:
            # Check if RSI was above 50 in the last 7 points before the crossover
            if any(rsi_series[-48:][max(0, i-7):i] > 50):
                crossovers.append({'index': data.index[-48:][i], 'type': 'bearish', 'rsi': rsi_series[-48:][i]})

            # Send Telegram message on bearish crossover
            message = f"Bearish Crossover Alert!\nDate/Time: {data.index[-48:][i]}\nRSI: {rsi_series[-48:][i]:.2f}"
            if not is_duplicate_message(message, message_log_df):
                print("Calling send_telegram_message with message:", message)  # Log statement
                send_telegram_message(message)
                message_log_df = add_message_to_log(message, message_log_df)
            else:
                print("Duplicate message suppressed:", message)  # Log statement

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
        x0=data.index[-48:][0], x1=data.index[-48:][-1], y0=50, y1=50, type="line",
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

# Telegram Bot Configuration

# Auto-refresh every 5 minutes
time.sleep(60)
st.rerun()