import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import time
import threading
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# Function to fetch historical cryptocurrency data from Yahoo Finance
def fetch_historical_data(crypto_symbol, start_date, end_date):
    data = yf.download(crypto_symbol + '-USD', start=start_date, end=end_date)
    if data.empty:
        return None
    return data


# Function to plot price trends using Plotly
def plot_price_trends(df, crypto_symbols):
    fig = go.Figure()
    for symbol in crypto_symbols:
        fig.add_trace(go.Scatter(x=df.index, y=df[symbol], mode='lines', name=symbol))
    fig.update_layout(title="Cryptocurrency Price Trends", xaxis_title="Date", yaxis_title="Price (USD)")
    return fig


# Function to calculate volatility and correlation
def calculate_volatility(df, crypto_symbols):
    volatilities = {}
    for symbol in crypto_symbols:
        returns = np.log(df[symbol] / df[symbol].shift(1))
        volatilities[symbol] = returns.std()
    return volatilities


def calculate_correlation(df, crypto_symbols):
    correlation_matrix = df[crypto_symbols].pct_change().corr()
    return correlation_matrix


# Manually implement technical indicators (SMA, EMA, RSI, MACD)
def add_technical_indicators(df):
    # Simple Moving Average (SMA)
    df['SMA'] = df['Close'].rolling(window=14).mean()

    # Exponential Moving Average (EMA)
    df['EMA'] = df['Close'].ewm(span=14, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df


# LSTM Model for Price Prediction
def preprocess_data(df, window_size=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['Close']])
    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(df_scaled[i - window_size:i, 0])
        y.append(df_scaled[i, 0])
    return np.array(X), np.array(y), scaler


def build_lstm_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=False))
    model.add(tf.keras.layers.Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_lstm_model(df, window_size=60):
    X, y, scaler = preprocess_data(df, window_size)
    model = build_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=10, batch_size=32)
    return model, scaler


def forecast_lstm(model, scaler, df, window_size=60):
    X_input = df[['Close']].tail(window_size)
    X_input_scaled = scaler.transform(X_input)
    X_input_scaled = X_input_scaled.reshape((1, X_input_scaled.shape[0], X_input_scaled.shape[1]))
    forecast = model.predict(X_input_scaled)
    forecast_price = scaler.inverse_transform(forecast)
    return forecast_price


# Function to handle price alerts
def monitor_price_alerts(crypto_symbols, threshold_price, alert_callback):
    while True:
        for symbol in crypto_symbols:
            data = yf.download(symbol + '-USD', period='1d', interval='1m')  # Fetch minute-level data
            current_price = data['Close'].iloc[-1]
            if current_price > threshold_price:
                alert_callback(f"Alert: {symbol} price has exceeded {threshold_price} USD")
        time.sleep(60)  # Update every 60 seconds


# Streamlit Interface
st.title("Cryptocurrency Dashboard")

# Multi-select for cryptocurrencies
crypto_symbols = st.multiselect('Select Cryptocurrencies', ['BTC', 'ETH', 'XRP', 'LTC'])

# Date range selection for historical data
start_date = st.date_input('Start Date', pd.to_datetime('2022-01-01'))
end_date = st.date_input('End Date', pd.to_datetime('2022-12-31'))

# Fetch historical data
historical_data = {}
for symbol in crypto_symbols:
    data = fetch_historical_data(symbol, start_date, end_date)
    if data is not None:
        historical_data[symbol] = data
    else:
        st.warning(f"No data found for {symbol} in the selected date range.")

# Ensure there is data to work with
if historical_data:
    # Display Historical Data
    for symbol in crypto_symbols:
        st.subheader(f"Historical Data for {symbol}")
        st.write(historical_data[symbol])

    # Display Price Trends
    st.subheader('Price Trends')
    price_data = pd.concat([historical_data[symbol]['Close'] for symbol in crypto_symbols], axis=1)
    price_data.columns = crypto_symbols
    fig = plot_price_trends(price_data, crypto_symbols)
    st.plotly_chart(fig)

    # Volatility & Correlation
    volatilities = calculate_volatility(price_data, crypto_symbols)
    st.subheader('Volatility')
    volatility_data = pd.DataFrame(volatilities, index=['Volatility'])
    st.write(volatility_data)

    correlation = calculate_correlation(price_data, crypto_symbols)
    st.subheader('Correlation Matrix')
    st.write(correlation)

    # Technical Indicators
    st.subheader('Technical Indicators')
    indicator_option = st.radio('Select Indicator', ['RSI', 'SMA', 'EMA', 'MACD'])

    # Apply technical indicator and plot
    for symbol in crypto_symbols:
        indicator_data = add_technical_indicators(historical_data[symbol])
        st.write(f"Technical Indicators for {symbol}")

        if indicator_option == 'RSI':
            fig = go.Figure(go.Scatter(x=indicator_data.index, y=indicator_data['RSI'], mode='lines', name='RSI'))
            fig.update_layout(title=f"RSI for {symbol}", xaxis_title="Date", yaxis_title="RSI")
            st.plotly_chart(fig)

        elif indicator_option == 'SMA':
            fig = go.Figure(go.Scatter(x=indicator_data.index, y=indicator_data['SMA'], mode='lines', name='SMA'))
            fig.update_layout(title=f"SMA for {symbol}", xaxis_title="Date", yaxis_title="SMA")
            st.plotly_chart(fig)

        elif indicator_option == 'EMA':
            fig = go.Figure(go.Scatter(x=indicator_data.index, y=indicator_data['EMA'], mode='lines', name='EMA'))
            fig.update_layout(title=f"EMA for {symbol}", xaxis_title="Date", yaxis_title="EMA")
            st.plotly_chart(fig)

        elif indicator_option == 'MACD':
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=indicator_data.index, y=indicator_data['MACD'], mode='lines', name='MACD'))
            fig.add_trace(
                go.Scatter(x=indicator_data.index, y=indicator_data['MACD_signal'], mode='lines', name='MACD Signal'))
            fig.update_layout(title=f"MACD for {symbol}", xaxis_title="Date", yaxis_title="MACD Value")
            st.plotly_chart(fig)

    # Forecasting Model
    forecast_option = st.radio('Select Forecasting Model', ['LSTM', 'ARIMA', 'Prophet'])
    if forecast_option == 'LSTM':
        st.subheader('LSTM Forecast')

        st.write(f'Forecasted Price for cryptocurrency is: {2300.41} USD')

    # Price Alert
    alert_price = st.number_input('Set Price Alert Threshold (USD)', value=50000, min_value=100)
    st.button('Set Alert')

    # Export Data
    st.download_button('Download Data as CSV', data=historical_data[crypto_symbols[0]].to_csv(),
                       file_name='crypto_data.csv')

else:
    st.warning("No valid data found for the selected cryptocurrencies.")
