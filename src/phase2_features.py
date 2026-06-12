# ============================================================
# PHASE 2: FEATURE ENGINEERING
# ============================================================
# We add technical indicators that traders use.
# These become the INPUT features for our ML models.

import pandas as pd
import numpy as np

def calculate_moving_averages(df):
    """
    Moving Average = average price over last N days.
    Used to smooth out noise and spot trends.
    """
    df["MA_20"] = df["Close"].rolling(window=20).mean()   # 20-day average
    df["MA_50"] = df["Close"].rolling(window=50).mean()   # 50-day average
    df["MA_200"] = df["Close"].rolling(window=200).mean() # 200-day average
    return df


def calculate_rsi(df, period=14):
    """
    RSI (Relative Strength Index) = measures if stock is
    overbought (>70) or oversold (<30). Range: 0 to 100.
    """
    delta = df["Close"].diff()  # daily change in price
    
    gain = delta.clip(lower=0)  # only positive changes
    loss = -delta.clip(upper=0) # only negative changes (made positive)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def calculate_macd(df):
    """
    MACD = difference between 12-day and 26-day EMA.
    Used to spot momentum and trend direction.
    Signal line = 9-day EMA of MACD.
    """
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()  # fast EMA
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()  # slow EMA
    
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def calculate_bollinger_bands(df, window=20):
    """
    Bollinger Bands = price channel based on volatility.
    Upper band = MA + 2*std, Lower band = MA - 2*std.
    When price hits upper band = overbought, lower = oversold.
    """
    rolling_mean = df["Close"].rolling(window=window).mean()
    rolling_std = df["Close"].rolling(window=window).std()
    
    df["BB_Upper"] = rolling_mean + (2 * rolling_std)
    df["BB_Lower"] = rolling_mean - (2 * rolling_std)
    df["BB_Mid"] = rolling_mean
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]  # how wide the band is
    return df


def calculate_volatility(df):
    """
    Daily Return = % change from yesterday to today.
    Volatility = how much the returns vary (std deviation).
    """
    df["Daily_Return"] = df["Close"].pct_change()  # e.g. 0.02 means +2%
    df["Volatility_20"] = df["Daily_Return"].rolling(20).std()  # 20-day volatility
    return df


def add_target_variable(df):
    """
    Target: Will the stock go UP tomorrow? (1=yes, 0=no)
    This is what our ML model will try to predict.
    """
    df["Next_Close"] = df["Close"].shift(-1)  # tomorrow's price
    df["Target_Direction"] = (df["Next_Close"] > df["Close"]).astype(int)
    # 1 = price went up next day, 0 = price went down
    return df


def engineer_features_for_one_stock(df):
    """
    Run all feature calculations for a single stock's data.
    Input: DataFrame with Date, Open, High, Low, Close, Volume columns
    Output: Same DataFrame with added feature columns
    """
    df = calculate_moving_averages(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    df = calculate_volatility(df)
    df = add_target_variable(df)
    
    # Remove rows where indicators couldn't be calculated (first N rows)
    feature_cols = ['MA_20', 'MA_50', 'MA_200', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Lower', 'BB_Mid', 'BB_Width', 'Daily_Return', 'Volatility_20', 'Next_Close', 'Target_Direction']
    df = df.dropna(subset=feature_cols)
    
    return df


# ---- MAIN: Apply to all stocks ----
if __name__ == "__main__":
    print("=== PHASE 2: Feature Engineering ===")
    
    # Load cleaned data from Phase 1
    raw_df = pd.read_csv("data/cleaned_data.csv", parse_dates=["Date"])
    
    all_featured = []
    
    for symbol in raw_df["Symbol"].unique():
        stock_df = raw_df[raw_df["Symbol"] == symbol].copy()
        featured = engineer_features_for_one_stock(stock_df)
        all_featured.append(featured)
        
    final_df = pd.concat(all_featured, ignore_index=True)
    
    print(f"\nFeature columns added: MA_20, MA_50, MA_200, RSI, MACD, BB_Upper, BB_Lower, etc.")
    print(f"Total rows with features: {len(final_df)}")
    
    # Save for ML phases
    final_df.to_csv("data/featured_data.csv", index=False)
    print("Phase 2 complete! Saved: data/featured_data.csv")
