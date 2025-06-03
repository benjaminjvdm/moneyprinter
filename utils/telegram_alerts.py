import streamlit as st
import asyncio
import telegram
import pandas as pd

# Utility function for sending Telegram alerts
def send_telegram_message(message):
    print("Inside send_telegram_message with message:", message)  # Log statement
    # WARNING: Hardcoding the token is not recommended. Use st.secrets instead.
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

# Create signal info table and send Telegram alerts for new signals
def create_ssl_cci_ema_signal_info(df, symbol):
    # Find the last buy or sell signal
    last_buy = df[df['buy_signal']].iloc[-1] if not df[df['buy_signal']].empty else None
    last_sell = df[df['sell_signal']].iloc[-1] if not df[df['sell_signal']].empty else None

    # Determine which one is more recent
    last_signal = None
    signal_type = None

    if last_buy is not None and last_sell is not None:
        if last_buy['Datetime'] > last_sell['Datetime']:
            last_signal = last_buy
            signal_type = "Buy"
        else:
            last_signal = last_sell
            signal_type = "Sell"
    elif last_buy is not None:
        last_signal = last_buy
        signal_type = "Buy"
    elif last_sell is not None:
        last_signal = last_sell
        signal_type = "Sell"

    if last_signal is not None:
        # Get the current price
        current_price = df['Close'].iloc[-1]

        # Calculate percentage change
        if signal_type == "Buy":
            pct_change = ((current_price - last_signal['Close']) / last_signal['Close']) * 100
        else:  # Sell
            pct_change = ((last_signal['Close'] - current_price) / last_signal['Close']) * 100

        # Format the datetime to show UTC+2
        signal_time = last_signal['Datetime'].strftime('%Y-%m-%d %H:%M:%S')

        # Check if this is a new signal that we haven't alerted on yet
        is_new_signal = False

        if signal_type == "Buy" and (st.session_state.last_buy_signal_time is None or
                                     last_signal['Datetime'] != st.session_state.last_buy_signal_time):
            st.session_state.last_buy_signal_time = last_signal['Datetime']
            is_new_signal = True
        elif signal_type == "Sell" and (st.session_state.last_sell_signal_time is None or
                                        last_signal['Datetime'] != st.session_state.last_sell_signal_time):
            st.session_state.last_sell_signal_time = last_signal['Datetime']
            is_new_signal = True

        # Send Telegram alert for new signals
        if is_new_signal:
            # Format the message with price, symbol, and strategy
            message = f"{signal_type} Signal Alert!\n" \
                      f"Symbol: {symbol}\n" \
                      f"Strategy: EMA CCI SSL\n" \
                      f"Date/Time: {signal_time}\n" \
                      f"Price: {last_signal['Close']:.2f}\n" \
                      f"RSI: {last_signal['RSI']:.2f}"

            # Send the message
            send_telegram_message(message)

        # Create a DataFrame for the signal info
        signal_info = pd.DataFrame({
            'Signal Type': [signal_type],
            'Signal Time (UTC+2)': [signal_time],
            'Signal Price': [last_signal['Close']],
            'Current Price': [current_price],
            'Change %': [f"{'+' if pct_change >= 0 else ''}{pct_change:.2f}%"]
        })

        return signal_info, pct_change >= 0

    return None, None