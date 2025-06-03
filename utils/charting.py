import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

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

# Create Plotly figure for RSI/EMA
def create_rsi_ema_chart(df):
    # Create RSI chart
    rsi_fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    # Add RSI trace
    rsi_fig.add_trace(
        go.Scatter(
            x=df['Datetime'],
            y=df['RSI'],
            name="RSI",
            line=dict(color="lightblue")
        )
    )

    # Add EMA of RSI trace
    rsi_fig.add_trace(
        go.Scatter(
            x=df['Datetime'],
            y=df['RSI_EMA'],
            name="RSI EMA",
            line=dict(color="hotpink")
        )
    )

    # Add RSI/EMA buy signals (crossover up and RSI > 50)
    rsi_buy_signals = df[df['rsi_cross_ema_up']]
    if not rsi_buy_signals.empty:
        rsi_fig.add_trace(
            go.Scatter(
                x=rsi_buy_signals['Datetime'],
                y=rsi_buy_signals['RSI_EMA'] - 5, # Slightly below EMA
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='green'),
                name='RSI/EMA Buy Signal'
            )
        )

    # Add RSI/EMA sell signals (crossover down and RSI < 50)
    rsi_sell_signals = df[df['rsi_cross_ema_down']]
    if not rsi_sell_signals.empty:
        rsi_fig.add_trace(
            go.Scatter(
                x=rsi_sell_signals['Datetime'],
                y=rsi_sell_signals['RSI_EMA'] + 5, # Slightly above EMA
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='red'),
                name='RSI/EMA Sell Signal'
            )
        )


    # Update layout
    rsi_fig.update_layout(
        xaxis_rangeslider_visible=False,
        yaxis=dict(range=[0, 100]),
        height=400,
        shapes = [dict(
            x0=df['Datetime'].iloc[0],
            x1=df['Datetime'].iloc[-1],
            y0=50, y1=50,
            type="line",
            line=dict(color="white", width=2, dash="dash")
        )]
    )

    # Format x-axis to show time in UTC+2
    rsi_fig.update_xaxes(
        tickformat="%Y-%m-%d %H:%M:%S",
        tickangle=-45
    )

    return rsi_fig