# ============================================================
# PHASE 4A: PORTFOLIO CONSTRUCTION
# ============================================================
# We build 3 portfolios for 3 types of investors:
#
# CONSERVATIVE: Low risk, steady returns (FD-beaters)
# BALANCED: Mix of safe and growth stocks
# AGGRESSIVE: High growth potential, higher risk

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

def select_stocks_for_profile(risk_df, profile):
    """
    Picks best stocks for each investor type:
    - Conservative: LOW RISK, good Sharpe
    - Balanced: Mix of LOW and MEDIUM RISK
    - Aggressive: MEDIUM and HIGH RISK with high returns
    """
    if profile == "Conservative":
        # Only low-risk stocks with Sharpe > 0.5
        candidates = risk_df[
            (risk_df["Risk_Level"] == "LOW RISK") &
            (risk_df["Sharpe_Ratio"] > 0.5)
        ]
        # Pick top 10 by Sharpe Ratio
        selected = candidates.nlargest(10, "Sharpe_Ratio")
        
    elif profile == "Balanced":
        # Mix: 60% low risk + 40% medium risk
        low = risk_df[risk_df["Risk_Level"] == "LOW RISK"].nlargest(6, "Sharpe_Ratio")
        medium = risk_df[risk_df["Risk_Level"] == "MEDIUM RISK"].nlargest(4, "Annual_Return")
        selected = pd.concat([low, medium])
        
    elif profile == "Aggressive":
        # Focus on high returns, accept more risk
        candidates = risk_df[
            (risk_df["Annual_Return"] > 0.10) |
            (risk_df["Risk_Level"] == "HIGH RISK")
        ]
        selected = candidates.nlargest(10, "Annual_Return")
    
    return selected


def equal_weight_portfolio(selected_stocks):
    """
    Simple approach: divide money equally among all selected stocks.
    e.g. if 10 stocks, put 10% in each.
    """
    n = len(selected_stocks)
    weights = {symbol: 1 / n for symbol in selected_stocks["Symbol"]}
    return weights


def sharpe_weighted_portfolio(selected_stocks):
    """
    Better approach: give more money to stocks with higher Sharpe ratio.
    Stocks that give better returns per unit risk get more allocation.
    """
    # Use only positive Sharpe ratios
    positive = selected_stocks[selected_stocks["Sharpe_Ratio"] > 0].copy()
    
    total_sharpe = positive["Sharpe_Ratio"].sum()
    
    weights = {}
    for _, row in positive.iterrows():
        weights[row["Symbol"]] = row["Sharpe_Ratio"] / total_sharpe
    
    return weights


def simulate_portfolio_returns(df, weights, start_date="2018-01-01"):
    """
    Given weights (how much in each stock), calculate 
    what the portfolio's value would have been over time.
    Starting with ₹1,00,000.
    """
    INITIAL_INVESTMENT = 100000  # Rs. 1 lakh
    
    portfolio_df = pd.DataFrame()
    
    for symbol, weight in weights.items():
        stock_df = df[df["Symbol"] == symbol][["Date", "Close"]].copy()
        stock_df = stock_df[stock_df["Date"] >= start_date]
        stock_df = stock_df.set_index("Date")
        
        # Normalize: what if we invested `weight * 1 lakh` here?
        initial_price = stock_df["Close"].iloc[0]
        stock_df[f"{symbol}_Value"] = (INITIAL_INVESTMENT * weight) * (stock_df["Close"] / initial_price)
        
        portfolio_df = pd.concat([portfolio_df, stock_df[[f"{symbol}_Value"]]], axis=1)
    
    # Total portfolio value each day
    portfolio_df["Total_Value"] = portfolio_df.sum(axis=1)
    
    return portfolio_df


def display_portfolio_summary(profile, weights, risk_df):
    """
    Print a nice summary of the portfolio.
    """
    print(f"\n{'='*50}")
    print(f"  {profile.upper()} INVESTOR PORTFOLIO")
    print(f"{'='*50}")
    print(f"{'Stock':<15} {'Allocation':>12} {'Sharpe':>10} {'Annual Return':>15}")
    print("-" * 55)
    
    for symbol, weight in sorted(weights.items(), key=lambda x: -x[1]):
        row = risk_df[risk_df["Symbol"] == symbol].iloc[0]
        print(f"{symbol:<15} {weight:>11.1%}  {row['Sharpe_Ratio']:>10.2f}  {row['Annual_Return']:>14.1%}")
    
    print(f"\nTotal stocks: {len(weights)}")


# ---- MAIN ----
if __name__ == "__main__":
    print("=== PHASE 4A: Portfolio Construction ===")
    
    risk_df = pd.read_csv("data/risk_profiles.csv")
    featured_df = pd.read_csv("data/featured_data.csv", parse_dates=["Date"])
    
    profiles = ["Conservative", "Balanced", "Aggressive"]
    all_portfolios = {}
    
    plt.figure(figsize=(14, 6))
    
    for profile in profiles:
        selected = select_stocks_for_profile(risk_df, profile)
        
        if len(selected) == 0:
            print(f"Not enough stocks for {profile} profile. Skipping.")
            continue
        
        weights = sharpe_weighted_portfolio(selected)
        display_portfolio_summary(profile, weights, risk_df)
        
        all_portfolios[profile] = weights
        
        # Simulate portfolio performance
        portfolio_returns = simulate_portfolio_returns(featured_df, weights)
        plt.plot(portfolio_returns.index, portfolio_returns["Total_Value"], 
                 label=f"{profile} (₹1L invested)", linewidth=2)
    
    plt.title("Portfolio Growth Over Time (Starting ₹1,00,000)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (₹)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/portfolio_comparison.png", dpi=150)
    plt.close()
    
    print("\nPhase 4A complete! Saved: outputs/portfolio_comparison.png")
