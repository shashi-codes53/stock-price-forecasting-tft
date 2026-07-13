"""
Data Pipeline for Stock Price Forecasting
Downloads NIFTY-50 stock data and computes technical indicators.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# NIFTY-50 top stocks
NIFTY50_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "AXISBANK.NS", "LT.NS", "WIPRO.NS", "HCLTECH.NS", "ASIANPAINT.NS",
]


# ─────────────────────────────────────────────
# 1. Technical Indicators
# ─────────────────────────────────────────────
def compute_rsi(series, period=14):
    """Relative Strength Index — momentum oscillator 0-100."""
    delta  = series.diff()
    gain   = delta.clip(lower=0).rolling(period).mean()
    loss   = (-delta.clip(upper=0)).rolling(period).mean()
    rs     = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    """MACD — trend-following momentum indicator."""
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line= macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(series, period=20, std_dev=2):
    """Bollinger Bands — volatility indicator."""
    sma   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower


def compute_atr(high, low, close, period=14):
    """Average True Range — volatility measure."""
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_technical_indicators(df):
    """
    Add all technical indicators to OHLCV dataframe.
    Returns dataframe with 16 features total.
    """
    close = df['Close']
    high  = df['High']
    low   = df['Low']
    vol   = df['Volume']

    # Price-based
    df['Returns']    = close.pct_change()
    df['SMA_10']     = close.rolling(10).mean()
    df['SMA_20']     = close.rolling(20).mean()
    df['SMA_50']     = close.rolling(50).mean()
    df['EMA_12']     = close.ewm(span=12, adjust=False).mean()

    # Momentum
    df['RSI']        = compute_rsi(close)
    macd, sig, hist  = compute_macd(close)
    df['MACD']       = macd
    df['MACD_Signal']= sig
    df['MACD_Hist']  = hist

    # Volatility
    bb_upper, bb_mid, bb_lower = compute_bollinger(close)
    df['BB_Upper']   = bb_upper
    df['BB_Lower']   = bb_lower
    df['BB_Width']   = (bb_upper - bb_lower) / bb_mid
    df['ATR']        = compute_atr(high, low, close)

    # Volume
    df['Volume_SMA'] = vol.rolling(20).mean()
    df['OBV']        = (np.sign(close.diff()) * vol).fillna(0).cumsum()

    return df


FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Returns', 'SMA_10', 'SMA_20', 'EMA_12', 'RSI',
    'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'BB_Width', 'ATR'
]


# ─────────────────────────────────────────────
# 2. Download Data
# ─────────────────────────────────────────────
def download_stock_data(ticker, period="3y"):
    """Download stock data using yfinance."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data for {ticker}")
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = add_technical_indicators(df)
        df = df.dropna()
        print(f"[Data] {ticker}: {len(df)} trading days loaded")
        return df
    except Exception as e:
        raise RuntimeError(f"Could not download {ticker}: {e}\nRun: pip install yfinance")


# ─────────────────────────────────────────────
# 3. Dataset
# ─────────────────────────────────────────────
class StockDataset(Dataset):
    """
    Sliding window dataset for time series forecasting.

    Each sample:
        X: seq_len days of features  → (seq_len, n_features)
        y: next forecast_horizon days of Close prices → (forecast_horizon,)
    """
    def __init__(self, df, seq_len=60, forecast_horizon=5, feature_cols=None):
        self.seq_len          = seq_len
        self.forecast_horizon = forecast_horizon
        feature_cols          = feature_cols or FEATURE_COLS

        # Scale features
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        X_raw = df[feature_cols].values
        y_raw = df[['Close']].values

        self.X_scaled = self.scaler_X.fit_transform(X_raw)
        self.y_scaled = self.scaler_y.fit_transform(y_raw).flatten()

        self.samples = []
        for i in range(len(df) - seq_len - forecast_horizon + 1):
            x_seq = self.X_scaled[i : i + seq_len]
            y_seq = self.y_scaled[i + seq_len : i + seq_len + forecast_horizon]
            self.samples.append((x_seq, y_seq))

        print(f"[Dataset] {len(self.samples)} samples | seq={seq_len} | horizon={forecast_horizon}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def inverse_transform_y(self, y_scaled):
        """Convert scaled predictions back to actual prices."""
        return self.scaler_y.inverse_transform(
            np.array(y_scaled).reshape(-1, 1)
        ).flatten()


def get_dataloaders(df, seq_len=60, forecast_horizon=5,
                    batch_size=32, val_split=0.15, test_split=0.1):
    """Split data and return train/val/test DataLoaders."""
    dataset = StockDataset(df, seq_len, forecast_horizon)
    n       = len(dataset)
    n_test  = int(n * test_split)
    n_val   = int(n * val_split)
    n_train = n - n_val - n_test

    train_ds = torch.utils.data.Subset(dataset, range(0, n_train))
    val_ds   = torch.utils.data.Subset(dataset, range(n_train, n_train + n_val))
    test_ds  = torch.utils.data.Subset(dataset, range(n_train + n_val, n))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print(f"[DataLoader] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, dataset
