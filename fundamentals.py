import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta


def calculate_technical_indicators(hist):
    """Calculate technical indicators"""
    # Moving averages
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    hist['MA200'] = hist['Close'].rolling(window=200).mean()

    # RSI
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    hist['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
    exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
    hist['MACD'] = exp1 - exp2
    hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()

    return hist


def get_stock_data(ticker):
    """Fetch stock data using yfinance"""
    stock = yf.Ticker(ticker)
    info = stock.info

    # Get historical data for the past year
    hist = stock.history(period="1y")
    hist = calculate_technical_indicators(hist)

    # Get financial statements
    balance_sheet = stock.balance_sheet
    income_stmt = stock.income_stmt
    cash_flow = stock.cash_flow

    return info, hist, balance_sheet, income_stmt, cash_flow


def calculate_metrics(info, hist, balance_sheet, income_stmt, cash_flow):
    """Calculate fundamental metrics"""
    metrics = {}

    # Price metrics
    metrics['Current Price'] = info.get('currentPrice', 0)
    metrics['52 Week High'] = info.get('fiftyTwoWeekHigh', 0)
    metrics['52 Week Low'] = info.get('fiftyTwoWeekLow', 0)
    metrics['Market Cap (B)'] = info.get('marketCap', 0) / 1e9

    # Valuation metrics
    metrics['P/E Ratio'] = info.get('trailingPE', 0)
    metrics['Forward P/E'] = info.get('forwardPE', 0)
    metrics['PEG Ratio'] = info.get('pegRatio', 0)
    metrics['Price/Book'] = info.get('priceToBook', 0)
    metrics['Price/Sales'] = info.get('priceToSalesTrailing12Months', 0)
    metrics['EV/EBITDA'] = info.get('enterpriseToEbitda', 0)

    # Financial health metrics
    metrics['Current Ratio'] = info.get('currentRatio', 0)
    metrics['Debt/Equity'] = info.get('debtToEquity', 0)
    metrics['Quick Ratio'] = info.get('quickRatio', 0)

    # Profitability metrics
    metrics['Return on Equity'] = info.get('returnOnEquity', 0)
    metrics['Return on Assets'] = info.get('returnOnAssets', 0)
    metrics['Profit Margin'] = info.get('profitMargins', 0)
    metrics['Operating Margin'] = info.get('operatingMargins', 0)
    metrics['Gross Margin'] = info.get('grossMargins', 0)

    # Growth metrics
    metrics['Revenue Growth'] = info.get('revenueGrowth', 0)
    metrics['Earnings Growth'] = info.get('earningsGrowth', 0)

    # Dividend metrics
    metrics['Dividend Yield'] = info.get('dividendYield', 0) if info.get('dividendYield') else 0
    metrics['Payout Ratio'] = info.get('payoutRatio', 0) if info.get('payoutRatio') else 0

    return metrics


def evaluate_stock(metrics):
    """Evaluate stock based on fundamental metrics"""
    score = 0
    max_score = 12
    reasons = []
    concerns = []

    # Valuation criteria
    if 0 < metrics['P/E Ratio'] < 25:
        score += 1
        reasons.append("P/E ratio is reasonable (< 25)")
    elif metrics['P/E Ratio'] > 35:
        concerns.append("High P/E ratio indicates potential overvaluation")

    if 0 < metrics['PEG Ratio'] < 1.5:
        score += 1
        reasons.append("PEG ratio indicates good value (< 1.5)")

    if 0 < metrics['Price/Book'] < 3:
        score += 1
        reasons.append("Price/Book ratio is attractive (< 3)")

    if 0 < metrics['EV/EBITDA'] < 15:
        score += 1
        reasons.append("EV/EBITDA indicates reasonable valuation (< 15)")

    # Financial health criteria
    if metrics['Current Ratio'] > 1.5:
        score += 1
        reasons.append("Strong current ratio (> 1.5)")
    elif metrics['Current Ratio'] < 1:
        concerns.append("Low current ratio indicates potential liquidity issues")

    if metrics['Debt/Equity'] < 1:
        score += 1
        reasons.append("Low debt-to-equity ratio (< 1)")
    elif metrics['Debt/Equity'] > 2:
        concerns.append("High debt levels relative to equity")

    # Profitability criteria
    if metrics['Return on Equity'] > 0.15:
        score += 1
        reasons.append("Strong Return on Equity (> 15%)")

    if metrics['Return on Assets'] > 0.07:
        score += 1
        reasons.append("Good Return on Assets (> 7%)")

    if metrics['Operating Margin'] > 0.15:
        score += 1
        reasons.append("Healthy operating margin (> 15%)")

    # Growth criteria
    if metrics['Revenue Growth'] > 0.1:
        score += 1
        reasons.append("Strong revenue growth (> 10%)")

    if metrics['Earnings Growth'] > 0.1:
        score += 1
        reasons.append("Strong earnings growth (> 10%)")

    # Dividend criteria
    if metrics['Dividend Yield'] > 0.02 and metrics['Payout Ratio'] < 0.75:
        score += 1
        reasons.append("Sustainable dividend with good yield (> 2%)")

    return score, max_score, reasons, concerns


def plot_technical_analysis(hist):
    """Create technical analysis charts"""
    # Create figure with secondary y-axis
    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True,  # Changed from shared_xaxis
                        vertical_spacing=0.05,
                        row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=('Price', 'RSI', 'MACD'))

    # Candlestick chart with MA
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close'],
        name='Price'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['MA50'],
        name='50-day MA',
        line=dict(color='orange')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['MA200'],
        name='200-day MA',
        line=dict(color='blue')
    ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['RSI'],
        name='RSI',
        line=dict(color='purple')
    ), row=2, col=1)

    # Add RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['MACD'],
        name='MACD',
        line=dict(color='blue')
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['Signal'],
        name='Signal',
        line=dict(color='orange')
    ), row=3, col=1)

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        title="Technical Analysis"
    )

    # Update y-axes labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    return fig


def calculate_technical_indicators(hist):
    """Calculate technical indicators"""
    # Moving averages
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    hist['MA200'] = hist['Close'].rolling(window=200).mean()

    # RSI
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    hist['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
    exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
    hist['MACD'] = exp1 - exp2
    hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()

    # Fill NaN values
    hist = hist.fillna(method='bfill')

    return hist


def get_stock_data(ticker):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get historical data for the past year
        hist = stock.history(period="1y")
        if hist.empty:
            raise ValueError(f"No historical data found for {ticker}")

        hist = calculate_technical_indicators(hist)

        # Get financial statements
        balance_sheet = stock.balance_sheet
        income_stmt = stock.income_stmt
        cash_flow = stock.cash_flow

        return info, hist, balance_sheet, income_stmt, cash_flow

    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        raise e


# [Rest of the code remains the same]

def main():
    st.set_page_config(layout="wide")
    st.title("Advanced Stock Analysis Dashboard")
    st.write("Enter a stock ticker to analyze its fundamentals and technical indicators")

    # User input
    ticker = st.text_input("Stock Ticker", "AAPL").upper()

    if st.button("Analyze"):
        try:
            with st.spinner('Fetching data...'):
                # Get stock data
                info, hist, balance_sheet, income_stmt, cash_flow = get_stock_data(ticker)
                metrics = calculate_metrics(info, hist, balance_sheet, income_stmt, cash_flow)

                # Company info
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.header(f"{info.get('longName', ticker)} ({ticker})")
                    st.write(info.get('longBusinessSummary', 'No description available'))

                with col2:
                    st.metric("Current Price", f"${metrics['Current Price']:.2f}")
                    st.metric("Market Cap", f"${metrics['Market Cap (B)']:.2f}B")

                # Technical Analysis
                st.plotly_chart(plot_technical_analysis(hist), use_container_width=True)

                # [Rest of the display code remains the same]

        except Exception as e:
            st.error(f"Error analyzing {ticker}. Please try again.")
            st.error(f"Error details: {str(e)}")


if __name__ == "__main__":
    main()