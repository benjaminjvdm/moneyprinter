import pandas as pd
import numpy as np
import pytz
from datetime import datetime

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