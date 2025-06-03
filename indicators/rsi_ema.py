import pandas as pd

# Logic for RSI EMA indicator
def calculate_rsi_ema_signals(df):
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Calculate EMA of RSI
    df['RSI_EMA'] = pd.Series(df['RSI']).ewm(span=14, adjust=False).mean()

    # Crossover/Crossunder logic with RSI level filter
    df['rsi_cross_ema_up'] = (df['RSI'] > df['RSI_EMA']) & (df['RSI'].shift(1) <= df['RSI_EMA'].shift(1)) & (df['RSI'] > 50)
    df['rsi_cross_ema_down'] = (df['RSI'] < df['RSI_EMA']) & (df['RSI'].shift(1) >= df['RSI_EMA'].shift(1)) & (df['RSI'] < 50)

    return df[['Datetime', 'RSI', 'RSI_EMA', 'rsi_cross_ema_up', 'rsi_cross_ema_down']]