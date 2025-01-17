import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
import base64

# Page Configuration
st.set_page_config(page_title="Advanced Stock Market Analytics", layout="wide")


# Helper Functions
def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a download link for a DataFrame or text object.
    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=True)
    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


@st.cache_data
def load_stock_data(symbol, period):
    """
    Fetches historical stock data from Yahoo Finance.
    """
    stock = yf.Ticker(symbol)
    data = stock.history(period=period)
    return data, stock.info


def calculate_technical_indicators(df):
    """
    Adds technical indicators to the DataFrame.
    """
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    return df


def plot_price_chart(df, show_sma, show_bb, show_volume):
    """
    Creates a candlestick chart with optional overlays.
    """
    fig = make_subplots(rows=3 if show_volume else 2, cols=1, shared_xaxes=True, vertical_spacing=0.03)

    # Candlestick
    fig.add_trace(
        go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="OHLC"),
        row=1, col=1,
    )

    # Simple Moving Averages
    if show_sma:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name="SMA 20", line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name="SMA 50", line=dict(color='orange')), row=1, col=1)

    # Bollinger Bands
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], name="BB High", line=dict(color='green')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], name="BB Low", line=dict(color='red')), row=1, col=1)

    # Volume
    if show_volume:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume"), row=2, col=1)

    fig.update_layout(height=600, showlegend=True, xaxis_rangeslider_visible=False)
    return fig


def predict_stock_prices(df, days_ahead, confidence=95):
    """
    Predicts future stock prices using RandomForestRegressor.
    """
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=20).std()

    # Feature engineering
    for i in range(1, 6):
        df[f'Lag_{i}'] = df['Close'].shift(i)
    df = df.dropna()

    # Preparing data
    X = df[['Returns', 'Volatility'] + [f'Lag_{i}' for i in range(1, 6)]]
    y = df['Close']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Forecasting
    last_row = X.iloc[-1]
    predictions, lower_bounds, upper_bounds = [], [], []

    z = norm.ppf(1 - (1 - confidence / 100) / 2)  # Z-score for confidence interval
    for _ in range(days_ahead):
        pred = model.predict(last_row.values.reshape(1, -1))[0]
        std = np.std([est.predict(last_row.values.reshape(1, -1))[0] for est in model.estimators_])
        predictions.append(pred)
        lower_bounds.append(pred - z * std)
        upper_bounds.append(pred + z * std)
        # Simulate a new row
        last_row = last_row.shift(-1)
        last_row[-1] = pred

    return predictions, lower_bounds, upper_bounds


# Sidebar Inputs
st.sidebar.header("Dashboard Settings")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
time_period = st.sidebar.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
show_sma = st.sidebar.checkbox("Show Moving Averages", True)
show_bb = st.sidebar.checkbox("Show Bollinger Bands", True)
show_volume = st.sidebar.checkbox("Show Volume", True)
prediction_days = st.sidebar.slider("Prediction Days", 5, 30, 7)
confidence_level = st.sidebar.slider("Confidence Interval (%)", 80, 99, 95)

# Main Layout
st.title("Advanced Stock Market Analytics Dashboard")

# Load and Display Data
try:
    data, info = load_stock_data(ticker, time_period)
    data = calculate_technical_indicators(data)

    # Display Stock Information
    st.subheader(f"{info['shortName']} ({info['symbol']})")
    st.write(f"Industry: {info.get('industry', 'N/A')}")
    st.write(f"Market Cap: {info.get('marketCap', 'N/A')}")
    st.write(f"52 Week High: {info.get('fiftyTwoWeekHigh', 'N/A')}")
    st.write(f"52 Week Low: {info.get('fiftyTwoWeekLow', 'N/A')}")

    # Display Charts
    st.subheader("Price Chart")
    fig = plot_price_chart(data, show_sma, show_bb, show_volume)
    st.plotly_chart(fig, use_container_width=True)

    # Predict Prices
    st.subheader("Price Predictions")
    preds, lowers, uppers = predict_stock_prices(data, prediction_days, confidence_level)
    future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=prediction_days, freq="B")

    pred_df = pd.DataFrame({"Date": future_dates, "Prediction": preds, "Lower Bound": lowers, "Upper Bound": uppers})
    st.write(pred_df)

    # Download Predictions
    st.markdown(download_link(pred_df, f"{ticker}_predictions.csv", "Download Predictions"), unsafe_allow_html=True)
except Exception as e:
    st.error(f"Error: {e}")
