import pandas as pd
import numpy as np
from utils.indicator_utils import calculate_cci

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