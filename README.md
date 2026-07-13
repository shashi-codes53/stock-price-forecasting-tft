---
title: Stock Forecast AI
emoji: 📈
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: streamlit_app.py
pinned: false
---

# 📈 Stock Price Forecasting with Temporal Fusion Transformer

Forecasts 5-day closing prices for NIFTY-50 stocks using a TFT model built from scratch in PyTorch.

## Features
- Live NIFTY-50 stock data via yFinance
- Candlestick chart with Bollinger Bands, SMA, EMA
- RSI and MACD subplots
- TFT model with LSTM + Multi-Head Attention
- 5-day price forecast with confidence visualization
- Buy / Sell / Hold trading signal
- Technical indicators summary

## Tech Stack
- PyTorch (TFT from scratch)
- Streamlit dashboard
- yFinance for live data
- Plotly for interactive charts
- Scikit-learn for feature scaling

## Run Locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Train Your Own Model
```bash
python train.py --ticker TCS.NS --epochs 50
```

## Built by
Shashikant Nikam — AI & Data Science, Sandip Institute of Technology
