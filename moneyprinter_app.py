import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go



def convert_to_utc_plus_2(dt):
    if isinstance(dt, pd.Timestamp):
        
        if dt.tzinfo is not None:
            
            return dt.astimezone(pytz.timezone('Etc/GMT-2'))
        else:
            
            utc_time = pytz.utc.localize(dt)
            return utc_time.astimezone(pytz.timezone('Etc/GMT-2'))
    return dt

def calculate_cci(df, n):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = tp.rolling(window=n).mean()
    md = tp.rolling(window=n).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    cci = (tp - ma) / (0.015 * md)
    return cci


def get_current_time():
    
    utc_now = datetime.now(pytz.utc)
    
    utc_plus_2 = utc_now.astimezone(pytz.timezone('Etc/GMT-2'))
    return utc_plus_2.strftime('%Y-%m-%d %H:%M:%S')


@st.cache_data(ttl=300)  
def get_data(ticker, interval, period):
    data = yf.download(ticker, interval=interval, period=period)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    
    data = data.reset_index()
    
    if 'Date' in data.columns:
        data.rename(columns={'Date': 'Datetime'}, inplace=True)
    
    
    if 'Datetime' in data.columns:
        data['Datetime'] = data['Datetime'].apply(convert_to_utc_plus_2)
    else:
        st.error("Error: 'Datetime' column not found in fetched data.")
        return pd.DataFrame() 
    return data




def create_ssl_cci_ema_chart(df, symbol):
    
    fig = go.Figure()

    
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

    
    
    reg_df = df.dropna(subset=['reg_mid', 'reg_upper', 'reg_lower'])

    if not reg_df.empty:
        
        fig.add_trace(
            go.Scatter(
                x=reg_df['Datetime'],
                y=reg_df['reg_mid'],
                name='Regression Mid',
                line=dict(color='blue', width=2, dash='solid')
            )
        )

        
        fig.add_trace(
            go.Scatter(
                x=reg_df['Datetime'],
                y=reg_df['reg_upper'],
                name='Regression Upper',
                line=dict(color='blue', width=1, dash='dash')
            )
        )

        
        fig.add_trace(
            go.Scatter(
                x=reg_df['Datetime'],
                y=reg_df['reg_lower'],
                name='Regression Lower',
                line=dict(color='blue', width=1, dash='dash')
            )
        )

    
    buy_signals = df[df['buy_signal']]
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals['Datetime'],
                y=buy_signals['Low'] * 0.999,  
                mode='markers+text',
                marker=dict(symbol='triangle-up', size=15, color='green'),
                text="Buy",
                textposition="bottom center",
                name='Buy Signal'
            )
        )

    
    sell_signals = df[df['sell_signal']]
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals['Datetime'],
                y=sell_signals['High'] * 1.001,  
                mode='markers+text',
                marker=dict(symbol='triangle-down', size=15, color='red'),
                text="Sell",
                textposition="top center",
                name='Sell Signal'
            )
        )

    
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

    
    fig.update_xaxes(
        tickformat="%Y-%m-%d %H:%M:%S",
        tickangle=-45
    )

    return fig


def calculate_ssl_cci_ema_signals(df):
    
    df = df.copy()

    
    
    htf_data = df.copy()
    
    htf_data = htf_data.set_index('Datetime')
    htf_data = htf_data.resample('1h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    
    htf_data['smaHigh'] = htf_data['High'].rolling(window=8).mean()
    htf_data['smaLow'] = htf_data['Low'].rolling(window=8).mean()

    
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

    
    htf_data['sslDown'] = np.where(htf_data['Hlv'] < 0, htf_data['smaHigh'], htf_data['smaLow'])
    htf_data['sslUp'] = np.where(htf_data['Hlv'] < 0, htf_data['smaLow'], htf_data['smaHigh'])

    
    htf_data = htf_data.reset_index()
    htf_data = htf_data[['Datetime', 'sslDown', 'sslUp']]

    
    df = df.set_index('Datetime')
    htf_data = htf_data.set_index('Datetime')

    
    df = pd.merge_asof(df.reset_index(), htf_data.reset_index(), on='Datetime', direction='backward')
    df = df.set_index('Datetime')
    df[['sslDown', 'sslUp']] = df[['sslDown', 'sslUp']].ffill()  

    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    
    df['RSI_EMA'] = pd.Series(df['RSI']).ewm(span=14, adjust=False).mean()

    rsiOverbought = 70
    rsiOversold = 30
    df['rsi_filter_buy'] = df['RSI'] < rsiOversold
    df['rsi_filter_sell'] = df['RSI'] > rsiOverbought

    
    
    df['cciTurbo'] = calculate_cci(df, 6)  
    df['cci14'] = calculate_cci(df, 14)    

    
    df['vol_ma'] = df['Volume'].rolling(window=20).mean()
    df['vol_spike'] = df['Volume'] > df['vol_ma'] * 1.5

    
    df['bullish_div'] = (df['Close'] < df['Close'].shift(1)) & (df['cciTurbo'] > df['cciTurbo'].shift(1))
    df['bearish_div'] = (df['Close'] > df['Close'].shift(1)) & (df['cciTurbo'] < df['cciTurbo'].shift(1))

    
    
    df['hammer'] = (df['Close'] > df['Open']) & ((df['Open'] - df['Low']) > 2 * (df['Close'] - df['Open']))

    
    df['shooting_star'] = (df['Open'] > df['Close']) & ((df['High'] - df['Open']) > 2 * (df['Open'] - df['Close']))

    
    df['enhanced_buy_signal'] = df['rsi_filter_buy'] & df['vol_spike'] & df['bullish_div'] & df['hammer']
    df['enhanced_sell_signal'] = df['rsi_filter_sell'] & df['vol_spike'] & df['bearish_div'] & df['shooting_star']

    
    df['slope_60'] = df['Close'] - df['Close'].shift(60)

    
    df['buy_condition'] = (df['slope_60'] < -2) & df['enhanced_buy_signal'] & (df['sslUp'] > df['sslDown']) & (df['Close'] > df['sslUp']) & (df['Close'].shift(1) < df['sslUp'].shift(1))
    df['sell_condition'] = (df['slope_60'] > 2) & df['enhanced_sell_signal'] & (df['sslUp'] < df['sslDown']) & (df['Close'] < df['sslUp']) & (df['Close'].shift(1) > df['sslUp'].shift(1))

    
    df['ssl_up_cross_down'] = (df['sslUp'] > df['sslDown']) & (df['sslUp'].shift(1) <= df['sslDown'].shift(1))
    df['ssl_down_cross_up'] = (df['sslUp'] < df['sslDown']) & (df['sslUp'].shift(1) >= df['sslDown'].shift(1))

    
    df['was_ssl_up_cross_down'] = df['ssl_up_cross_down'].shift(1).fillna(False)
    df['was_ssl_down_cross_up'] = df['ssl_down_cross_up'].shift(1).fillna(False)

    df['buy_signal'] = df['ssl_up_cross_down'] & ~df['was_ssl_up_cross_down']
    df['sell_signal'] = df['ssl_down_cross_up'] & ~df['was_ssl_down_cross_up']

    
    
    df['reg_mid'] = np.nan
    df['reg_upper'] = np.nan
    df['reg_lower'] = np.nan
    df['reg_slope'] = np.nan

    
    length = 60
    devlen = 2.0

    if len(df) >= length:
        for i in range(length, len(df)):
            
            window_data = df['Close'].iloc[i-length:i].values
            x = np.arange(length)

            
            try:
                slope, intercept = np.polyfit(x, window_data, 1)

                
                reg_line = intercept + slope * x

                
                dev = np.sqrt(np.sum((window_data - reg_line)**2) / length)

                
                midline = reg_line[-1]  
                upper_line = midline + devlen * dev
                lower_line = midline - devlen * dev

                
                df.iloc[i, df.columns.get_loc('reg_mid')] = midline
                df.iloc[i, df.columns.get_loc('reg_upper')] = upper_line
                df.iloc[i, df.columns.get_loc('reg_lower')] = lower_line
                df.iloc[i, df.columns.get_loc('reg_slope')] = slope
            except:
                
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


st.set_page_config(
    page_title="EMA CCI SSL BUY SELL Signal [THANHCONG]",
    layout="wide"
)


st.title("EMA CCI SSL BUY SELL Signal [THANHCONG]")
st.markdown("Python version of TradingView PineScript indicator")


symbol = "GC=F"
interval = "5m" 


if 'last_buy_signal_time' not in st.session_state:
    st.session_state.last_buy_signal_time = None
if 'last_sell_signal_time' not in st.session_state:
    st.session_state.last_sell_signal_time = None


if st.button("Refresh Charts"):
    st.session_state.last_refresh = get_current_time()
    st.rerun()


if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = get_current_time()


col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Chart: {symbol} (4h) - UTC+2 Timezone")
    data_4h_raw = get_data(symbol, "1h", "60d") 
    if data_4h_raw.empty:
        st.error(f"No data available for {symbol} (4h)")
    else:
        
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
    data_1d = get_data(symbol, "1d", "1y") 
    if data_1d.empty:
        st.error(f"No data available for {symbol} (1d)")
    else:
        df_ssl_cci_ema_1d = calculate_ssl_cci_ema_signals(data_1d.copy())
        chart_ssl_cci_ema_1d = create_ssl_cci_ema_chart(df_ssl_cci_ema_1d, symbol)
        st.plotly_chart(chart_ssl_cci_ema_1d, use_container_width=True)


st.write(f"Last updated: {st.session_state.last_refresh}")