import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(
    page_title="ProTrade • AI Stock Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------
# Custom CSS & Aesthetics
# ----------------------
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        color: #FAFAFA;
        font-size: 3rem;
        margin-bottom: 0px;
    }
    .sub-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 400;
        color: #A0A0A0;
        font-size: 1.2rem;
        margin-bottom: 20px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #00ADB5;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: #DDDDDD;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    /* Card-like styling for containers */
    div.css-1r6slb0 {
        border: 1px solid #333;
        padding: 20px;
        border-radius: 10px;
        background-color: #161B22;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ----------------------
# Sidebar (Controls)
# ----------------------
with st.sidebar:
    st.title("⚡ ProTrade Control")
    ticker = st.text_input("Stock Ticker", value="RELIANCE.NS", help="e.g., AAPL, GOOGL, RELIANCE.NS")
    period = st.selectbox("Historical Data", ["1y", "2y", "5y", "10y"], index=2)
    future_days = st.slider("Forecast Horizon (Days)", 1, 30, 7)
    
    st.markdown("---")
    st.write("### Model Settings")
    epochs = st.slider("Training Epochs", 10, 100, 25)
    batch_size = st.slider("Batch Size", 16, 64, 32)
    
    if st.button("🚀 Run Prediction", type="primary"):
        st.session_state.run = True

# ----------------------
# Data Processing Class
# ----------------------
class StockDataProcessor:
    def __init__(self, ticker, period):
        self.ticker = ticker
        self.period = period
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    @st.cache_data(ttl=3600)
    def fetch_data(_self):
        data = yf.download(_self.ticker, period=_self.period, progress=False)
        # Handle MultiIndex columns (yfinance v0.2+ can return this even for single ticker)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data

    def add_indicators(self, df):
        df = df.copy()
        # Ensure we have data
        if df.empty:
            return df
            
        # 1. Moving Averages
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # 2. RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 4. Bollinger Bands
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['STD20'] = df['Close'].rolling(window=20).std()
        df['Upper_BB'] = df['MA20'] + (df['STD20'] * 2)
        df['Lower_BB'] = df['MA20'] - (df['STD20'] * 2)
        
        # 5. Volume (OBV approximation or just raw volume)
        # Using raw volume as a feature is good for LSTM
        
        df.dropna(inplace=True)
        return df

    def get_sentiment(self):
        # NOTE: This is for display only. Not safe for historical training without back-filled data.
        analyzer = SentimentIntensityAnalyzer()
        news_url = f"https://in.finance.yahoo.com/quote/{self.ticker}/news?p={self.ticker}"
        sentiments = []
        try:
            response = requests.get(news_url)
            soup = BeautifulSoup(response.text, "html.parser")
            headlines = soup.find_all("h3")[:10]  # First 10 headlines
            for item in headlines:
                text = item.text
                score = analyzer.polarity_scores(text)['compound']
                sentiments.append((text, score))
        except Exception as e:
            return [], 0
            
        avg_score = np.mean([s[1] for s in sentiments]) if sentiments else 0
        return sentiments, avg_score

# ----------------------
# Main App Logic
# ----------------------
st.markdown('<div class="main-header">ProTrade AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Technical Analysis & LSTM Forecasting</div>', unsafe_allow_html=True)

processor = StockDataProcessor(ticker, period)
data = processor.fetch_data()

if data.empty:
    st.error(f"No data found for {ticker}. Check symbol.")
    st.stop()

data = processor.add_indicators(data)

# --- Dashboard Top Row ---
col1, col2, col3, col4 = st.columns(4)

# Helper to get scalar value safely
def get_scalar(val):
    if isinstance(val, pd.Series):
        return val.item()
    return val

current_price = get_scalar(data['Close'].iloc[-1])
prev_price = get_scalar(data['Close'].iloc[-2])
price_change = current_price - prev_price
pct_change = (price_change / prev_price) * 100

with col1:
    st.metric("Current Price", f"{current_price:.2f}", f"{pct_change:.2f}%")
with col2:
    rsi_val = get_scalar(data['RSI'].iloc[-1])
    st.metric("RSI (14)", f"{rsi_val:.2f}", delta_color="off")
with col3:
    macd_val = get_scalar(data['MACD'].iloc[-1])
    st.metric("MACD", f"{macd_val:.2f}", delta_color="off")
with col4:
    vol_val = get_scalar(data['Volume'].iloc[-1])
    st.metric("Volume", f"{vol_val:,.0f}")

# --- Tabs ---
tab_chart, tab_forecast, tab_news = st.tabs(["📊 Technical Chart", "🔮 AI Forecast", "📰 Live Sentiment"])

# 1. Technical Chart Tab
with tab_chart:
    st.subheader(f"{ticker} - Market Overview")
    
    # Create interactive Plotly Candlestick chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=('Price Action', 'Volume'), 
                        row_width=[0.2, 0.7])

    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'], high=data['High'],
                                 low=data['Low'], close=data['Close'], name="OHLC"), 
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_10'], line=dict(color='orange', width=1), name="EMA 10"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_50'], line=dict(color='blue', width=1), name="EMA 50"), row=1, col=1)
    
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], marker_color='rgba(0, 200, 200, 0.5)', name="Volume"), 
                  row=2, col=1)

    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark",
                      margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

# 2. AI Forecast Tab
with tab_forecast:
    st.subheader(f"LSTM Prediction Model ({future_days} Days Out)")
    
    if st.button("Start Training & Predict") or st.session_state.get('run'):
        
        # Prepare Data for LSTM
        # Using Features: Close, RSI, MACD, Signal_Line, Volume, EMA_10
        feature_cols = ['Close', 'RSI', 'MACD', 'Signal_Line', 'Volume']
        dataset = data[feature_cols].values
        
        # Scale Data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        
        # Create Sequences
        x_train, y_train = [], []
        seq_len = 60
        
        # We predict 'Close' which is at index 0
        prediction_col_index = 0 

        # Build training sequences
        for i in range(seq_len, len(scaled_data) - future_days):
            x_train.append(scaled_data[i-seq_len:i])
            y_train.append(scaled_data[i+future_days, prediction_col_index])
            
        x_train, y_train = np.array(x_train), np.array(y_train)
        
        # LSTM Architecture
        tf.keras.backend.clear_session()
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Progress Bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Custom basic 'callback' via manual loop provided Keras verbose=0 is used or we just wait
        status_text.text("Training Neural Network...")
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        progress_bar.progress(100)
        status_text.text("Training Complete!")
        
        # Prediction
        # Get the last sequence (most recent data)
        last_sequence = scaled_data[-seq_len:]
        last_sequence = last_sequence.reshape(1, seq_len, len(feature_cols))
        
        predicted_scaled = model.predict(last_sequence)
        
        # Inverse transform
        # We need to construct a dummy array to inverse transform because scaler expects 5 features
        dummy_array = np.zeros((1, len(feature_cols)))
        dummy_array[0, 0] = predicted_scaled[0][0] # Set Close Price
        predicted_price = scaler.inverse_transform(dummy_array)[0][0]
        
        # Display Result
        st.success(f"### Predicted Price in {future_days} days: **{predicted_price:.2f}**")
        
        # Visualization of Forecast
        # Simple projection: Draw a line from current price to predicted price
        last_date = data.index[-1]
        future_date = last_date + pd.Timedelta(days=future_days)
        
        fig_pred = go.Figure()
        
        # Historical (Last 90 days for clarity)
        subset = data.iloc[-90:]
        fig_pred.add_trace(go.Scatter(x=subset.index, y=subset['Close'], 
                                      mode='lines', name='Historical Close', line=dict(color='#00ADB5', width=2)))
        
        # Forecast Line
        fig_pred.add_trace(go.Scatter(x=[last_date, future_date], 
                                      y=[subset['Close'].iloc[-1], predicted_price],
                                      mode='lines+markers', name='Forecast', 
                                      line=dict(color='#FF2E63', width=3, dash='dot'),
                                      marker=dict(size=10)))
                                      
        fig_pred.update_layout(title="Short-Term Price Forecast", template="plotly_dark", height=500)
        st.plotly_chart(fig_pred, use_container_width=True)
        
    else:
        st.info("👈 Click 'Run Prediction' in the sidebar or above to start the AI model.")
    
# 3. Sentiment Tab
with tab_news:
    st.subheader("Live News & Sentiment Analysis")
    st.caption("Note: This sentiment is derived from today's headlines and is used for market context, not historical training.")
    
    sentiments, avg_score = processor.get_sentiment()
    
    if sentiments:
        # Gauge for Sentiment
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = avg_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Market Sentiment Score (-1 to +1)"},
            gauge = {'axis': {'range': [-1, 1]},
                     'bar': {'color': "white"},
                     'steps': [
                         {'range': [-1, -0.05], 'color': "#FF2E63"},
                         {'range': [-0.05, 0.05], 'color': "gray"},
                         {'range': [0.05, 1], 'color': "#00ADB5"}],
                     'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': avg_score}}))
        
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown("### Top Headlines")
        for text, score in sentiments:
            color = "#00ADB5" if score > 0.05 else "#FF2E63" if score < -0.05 else "#A0A0A0"
            st.markdown(f"<span style='color:{color}; font-weight:bold'>[{score:.2f}]</span> {text}", unsafe_allow_html=True)
            st.divider()
    else:
        st.warning("Could not fetch news at this time.")

