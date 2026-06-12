# ============================================================
# PHASE 4B: ANOMALY DETECTION
# ============================================================
# We look for:
# 1. Extreme price moves (days when stock moved way more than usual)
# 2. Unusual volume spikes (heavy buying/selling)
# 3. Maximum drawdown periods (market crashes)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

def detect_extreme_returns(df, symbol, threshold=3.0):
    """
    Finds days when the stock moved more than 3 standard deviations.
    These are statistically unusual events.
    threshold=3.0 means "3 times the normal daily movement"
    """
    stock_df = df[df["Symbol"] == symbol].sort_values("Date").copy()
    
    returns = stock_df["Close"].pct_change()
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Z-score: how many standard deviations away from normal?
    stock_df["Return_ZScore"] = (returns - mean_return) / std_return
    
    # Flag extreme days
    stock_df["Is_Anomaly"] = stock_df["Return_ZScore"].abs() > threshold
    
    anomalies = stock_df[stock_df["Is_Anomaly"]][["Date", "Close", "Daily_Return", "Return_ZScore"]]
    
    return stock_df, anomalies


def detect_volume_spikes(df, symbol, threshold=2.5):
    """
    Finds days when trading volume was 2.5x higher than usual.
    High volume often signals important news or institutional activity.
    """
    stock_df = df[df["Symbol"] == symbol].sort_values("Date").copy()
    
    avg_volume = stock_df["Volume"].rolling(20).mean()
    stock_df["Volume_Ratio"] = stock_df["Volume"] / avg_volume
    
    volume_spikes = stock_df[stock_df["Volume_Ratio"] > threshold][
        ["Date", "Volume", "Volume_Ratio", "Close"]
    ]
    
    return volume_spikes


def plot_anomalies(df, symbol):
    """
    Creates a chart showing stock price with anomalies marked in red.
    """
    stock_df, anomalies = detect_extreme_returns(df, symbol)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Price chart with anomalies marked
    ax1.plot(stock_df["Date"], stock_df["Close"], color="steelblue", linewidth=1, label="Close Price")
    ax1.scatter(anomalies["Date"], anomalies["Close"], 
                color="red", s=30, zorder=5, label="Anomaly (extreme move)")
    ax1.set_title(f"Price with Anomalies: {symbol}")
    ax1.set_ylabel("Price (INR)")
    ax1.legend()
    
    # Z-score chart
    ax2.plot(stock_df["Date"], stock_df["Return_ZScore"], color="gray", linewidth=0.8)
    ax2.axhline(y=3, color="red", linestyle="--", alpha=0.7, label="+3σ threshold")
    ax2.axhline(y=-3, color="red", linestyle="--", alpha=0.7, label="-3σ threshold")
    ax2.set_title("Daily Return Z-Score")
    ax2.set_ylabel("Z-Score")
    ax2.set_xlabel("Date")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"outputs/anomalies_{symbol}.png", dpi=150)
    plt.close()
    print(f"Saved: outputs/anomalies_{symbol}.png")


# ---- MAIN ----
if __name__ == "__main__":
    print("=== PHASE 4B: Anomaly Detection ===")
    
    df = pd.read_csv("data/featured_data.csv", parse_dates=["Date"])
    
    # Analyze top 5 stocks
    for symbol in ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]:
        if symbol in df["Symbol"].values:
            stock_df, anomalies = detect_extreme_returns(df, symbol)
            vol_spikes = detect_volume_spikes(df, symbol)
            
            print(f"\n{symbol}:")
            print(f"  Extreme price moves detected: {len(anomalies)}")
            print(f"  Volume spikes detected: {len(vol_spikes)}")
            
            plot_anomalies(df, symbol)
    
    print("\nPhase 4B complete!")
