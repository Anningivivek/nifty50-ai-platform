# ============================================================
# PHASE 3B: RISK ASSESSMENT MODULE
# ============================================================
# Sharpe Ratio: return vs risk (higher = better)
# Max Drawdown: worst possible loss from peak to trough
# Volatility: how wildly the price swings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

RISK_FREE_RATE = 0.06  # 6% annual return from government bonds (India)

def calculate_sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    """
    Sharpe Ratio = (Average Return - Risk Free Rate) / Std Deviation
    Tells us: how much return are we getting per unit of risk?
    > 1.0 is good, > 2.0 is excellent
    """
    daily_rf = risk_free_rate / 252  # convert annual to daily
    excess_returns = returns - daily_rf
    
    if returns.std() == 0:
        return 0
    
    # Multiply by sqrt(252) to annualize
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    return round(sharpe, 4)


def calculate_sortino_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    """
    Sortino Ratio = like Sharpe but only penalizes downside risk.
    More relevant because we don't mind upside volatility!
    """
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    
    if downside_std == 0:
        return 0
    
    sortino = (excess_returns.mean() / downside_std) * np.sqrt(252)
    return round(sortino, 4)


def calculate_max_drawdown(prices):
    """
    Max Drawdown = biggest loss from a peak to a trough.
    e.g. -0.40 means the stock fell 40% from its highest point.
    """
    # Running maximum (peak so far)
    rolling_max = prices.cummax()
    
    # How far below the peak are we at each point?
    drawdown = (prices - rolling_max) / rolling_max
    
    max_dd = drawdown.min()  # worst drawdown ever
    return round(max_dd, 4)


def calculate_var(returns, confidence=0.95):
    """
    VaR = Value at Risk: 
    If confidence=0.95, this is the loss you could expect 
    on your worst 5% of days.
    e.g. -0.03 means on bad days you might lose 3%
    """
    var = np.percentile(returns.dropna(), (1 - confidence) * 100)
    return round(var, 4)


def get_risk_profile(df, symbol):
    """
    Calculate all risk metrics for one stock.
    Returns a dictionary of metrics.
    """
    stock_df = df[df["Symbol"] == symbol].sort_values("Date").copy()
    
    prices = stock_df["Close"]
    returns = prices.pct_change().dropna()
    
    # Annual return
    total_days = len(returns)
    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / total_days) - 1
    
    metrics = {
        "Symbol": symbol,
        "Annual_Return": round(annual_return, 4),
        "Annual_Volatility": round(returns.std() * np.sqrt(252), 4),
        "Sharpe_Ratio": calculate_sharpe_ratio(returns),
        "Sortino_Ratio": calculate_sortino_ratio(returns),
        "Max_Drawdown": calculate_max_drawdown(prices),
        "VaR_95": calculate_var(returns),
    }
    return metrics


def classify_risk_level(metrics):
    """
    Based on metrics, classify each stock as:
    - LOW RISK: Sharpe > 0.6
    - MEDIUM RISK: Sharpe > 0.3
    - HIGH RISK: Sharpe <= 0.3
    """
    sharpe = metrics["Sharpe_Ratio"]
    
    # Classify Risk Profile based on Sharpe Ratio
    if sharpe > 0.6:
        return "LOW RISK"
    elif sharpe > 0.3:
        return "MEDIUM RISK"
    else:
        return "HIGH RISK"


# ---- MAIN ----
if __name__ == "__main__":
    print("=== PHASE 3B: Risk Assessment ===")
    
    df = pd.read_csv("data/featured_data.csv", parse_dates=["Date"])
    
    all_risk_profiles = []
    
    for symbol in df["Symbol"].unique():
        try:
            profile = get_risk_profile(df, symbol)
            profile["Risk_Level"] = classify_risk_level(profile)
            all_risk_profiles.append(profile)
        except Exception as e:
            print(f"Could not compute risk for {symbol}: {e}")
    
    risk_df = pd.DataFrame(all_risk_profiles)
    
    # Sort by Sharpe Ratio (best to worst)
    risk_df = risk_df.sort_values("Sharpe_Ratio", ascending=False)
    
    print("\n=== TOP 10 STOCKS BY SHARPE RATIO ===")
    print(risk_df[["Symbol", "Annual_Return", "Sharpe_Ratio", "Max_Drawdown", "Risk_Level"]].head(10).to_string(index=False))
    
    # Plot: Sharpe vs Annual Return (risk-return scatter)
    plt.figure(figsize=(10, 6))
    colors = {"LOW RISK": "green", "MEDIUM RISK": "orange", "HIGH RISK": "red"}
    for risk_level, group in risk_df.groupby("Risk_Level"):
        plt.scatter(group["Annual_Volatility"], group["Annual_Return"],
                    label=risk_level, color=colors[risk_level], s=60, alpha=0.7)
        for _, row in group.iterrows():
            plt.annotate(row["Symbol"], (row["Annual_Volatility"], row["Annual_Return"]),
                         fontsize=7, alpha=0.7)
    
    plt.xlabel("Annual Volatility (Risk)")
    plt.ylabel("Annual Return")
    plt.title("Risk vs Return: NIFTY-50 Stocks")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/risk_return_scatter.png", dpi=150)
    plt.close()
    
    # Save results
    risk_df.to_csv("data/risk_profiles.csv", index=False)
    print("\nPhase 3B complete! Saved: data/risk_profiles.csv")
