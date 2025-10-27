![Daily Refresh](https://github.com/yubyunghun/stock-direction-ml/actions/workflows/daily_refresh.yml/badge.svg)

# Stock / Crypto Direction Classifier

Predict next-day price **direction (up/down)** using stock/crypto OHLCV data.  
Features include technical indicators, rolling statistics, and deep learning (LSTM/GRU).

## Project Plan
- Data: Yahoo Finance, Binance CSV
- Features: Returns, volatility, RSI, MACD, SMA/EMA, Bollinger
- Models: Logistic Regression, XGBoost, LSTM/GRU
- Backtest: Vectorized simulation with transaction costs
- Demo: Streamlit app to visualize signals & equity curves

⚠️ **Disclaimer**: This is an educational project — not financial advice.
