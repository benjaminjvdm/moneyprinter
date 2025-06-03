import yfinance as yf
import pandas as pd
import streamlit as st
from utils.indicator_utils import convert_to_utc_plus_2

# Utility function for fetching data
@st.cache_data(ttl=300)  # Cache data for 5 minutes
def get_data(ticker, interval, period):
    data = yf.download(ticker, interval=interval, period=period)
    # Flatten multi-index columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    # Convert index to UTC+2
    data = data.reset_index()
    data['Datetime'] = data['Datetime'].apply(convert_to_utc_plus_2)
    return data