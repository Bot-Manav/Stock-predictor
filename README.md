# ProTrade • AI Stock Predictor ⚡

ProTrade is an advanced AI-powered stock prediction and technical analysis dashboard built with Streamlit. It fetches real-time stock data, performs technical analysis, analyzes news sentiment, and uses Long Short-Term Memory (LSTM) neural networks to forecast future stock prices.

## Features 🚀

-   **Real-time Data**: Fetches live stock data using `yfinance`.
-   **Technical Analysis**:
    -   Candlestick charts with EMA (10 & 50 period).
    -   RSI (Relative Strength Index).
    -   MACD (Moving Average Convergence Divergence).
    -   Bollinger Bands.
    -   Volume analysis.
-   **AI Forecasting**:
    -   Built-in LSTM (Deep Learning) model trained on-the-fly.
    -   Customizable training parameters (Epochs, Batch Size).
    -   Predicts stock price for a specified future horizon (1-30 days).
-   **Sentiment Analysis**:
    -   Scrapes live news headlines from Yahoo Finance.
    -   Uses VADER Sentiment Analysis to gauge market mood.

## Prerequisites 🛠️

Ensure you have Python 3.7+ installed. Install the required dependencies using pip:

```bash
pip install streamlit yfinance pandas numpy requests beautifulsoup4 vaderSentiment tensorflow scikit-learn plotly
```

## How to Run ▶️

1.  Clone this repository or download `stock_predict.py`.
2.  Navigate to the directory containing the file.
3.  Run the Streamlit app:

```bash
streamlit run stock_predict.py
```

## Usage 📖

1.  **Sidebar Controls**:
    -   **Stock Ticker**: Enter a valid ticker symbol (e.g., `RELIANCE.NS` for NSE, `AAPL` for NASDAQ).
    -   **Historical Data**: Select the lookup period (1y, 2y, 5y, 10y).
    -   **Forecast Horizon**: Choose how many days into the future you want to predict.
    -   **Model Settings**: Adjust Training Epochs and Batch Size for the LSTM model.
2.  **Dashboard**: View key metrics (Current Price, RSI, MACD, Volume).
3.  **Tabs**:
    -   **Technical Chart**: Interact with the candlestick chart.
    -   **AI Forecast**: Click "Start Training & Predict" to train the model and see the forecast.
    -   **Live Sentiment**: Check real-time news sentiment scores.

## Disclaimer ⚠️

This tool is for educational and research purposes only. **Do not use this for actual financial trading.** Stock market prediction involves significant risk, and AI models can be inaccurate. Always do your own research.

## Copyright 🔒

Copyright © 2025 Manav. All Rights Reserved.

This software and associated documentation files (the "Software") are the proprietary property of Manav. 
**No one else is allowed to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.** 
Unauthorized use is strictly prohibited.
