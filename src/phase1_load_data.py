# ============================================================
# PHASE 1: DATA LOADING AND EDA
# ============================================================
# This phase loads all 50 stock CSV files, cleans the data,
# and creates basic charts to understand what we have.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---- STEP 1.1: Load all CSV files ----

DATA_FOLDER = "data/"   # Put your Kaggle CSV files here

def load_all_stocks(data_folder):
    """
    Loads every CSV file in the data folder.
    Returns a single big DataFrame with a 'Symbol' column.
    """
    all_dfs = []
    
    for filename in os.listdir(data_folder):
        if filename.endswith(".csv"):
            symbol = filename.replace(".csv", "")  # e.g. "RELIANCE"
            filepath = os.path.join(data_folder, filename)
            
            df = pd.read_csv(filepath)
            df["Symbol"] = symbol  # add which company this is
            all_dfs.append(df)
    
    # Combine all companies into one big table
    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


# ---- STEP 1.2: Clean the data ----

def clean_data(df):
    """
    Cleans the raw data:
    - Converts Date column to proper date type
    - Removes rows with missing prices
    - Sorts by Symbol and Date
    """
    # Convert date to proper format
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Remove rows where Close price is missing
    df = df.dropna(subset=["Close"])
    
    # Sort by company and date
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    
    # Rename columns to standard names (adjust if your CSV uses different names)
    # Common Kaggle column names: Date, Open, High, Low, Close, Volume, Turnover
    
    print(f"Total rows loaded: {len(df)}")
    print(f"Total companies: {df['Symbol'].nunique()}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"\nColumns available: {list(df.columns)}")
    
    return df


# ---- STEP 1.3: Basic EDA Charts ----

def run_eda(df):
    """
    Creates basic charts to understand our data.
    Saves them to the outputs/ folder.
    """
    os.makedirs("outputs", exist_ok=True)
    
    # Chart 1: Price history of top 5 companies
    top_symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    
    plt.figure(figsize=(14, 6))
    for symbol in top_symbols:
        stock_df = df[df["Symbol"] == symbol]
        if not stock_df.empty:
            plt.plot(stock_df["Date"], stock_df["Close"], label=symbol, linewidth=1.5)
    
    plt.title("Stock Price History - Top 5 NIFTY Companies")
    plt.xlabel("Year")
    plt.ylabel("Closing Price (INR)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/price_history.png", dpi=150)
    plt.close()
    print("Saved: outputs/price_history.png")
    
    # Chart 2: Trading volume by company (average)
    avg_volume = df.groupby("Symbol")["Volume"].mean().sort_values(ascending=False).head(15)
    
    plt.figure(figsize=(12, 5))
    avg_volume.plot(kind="bar", color="steelblue")
    plt.title("Average Daily Trading Volume (Top 15 Companies)")
    plt.ylabel("Average Volume")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/avg_volume.png", dpi=150)
    plt.close()
    print("Saved: outputs/avg_volume.png")
    
    # Chart 3: Correlation heatmap of closing prices
    # Pivot: rows = date, columns = symbol, values = close price
    pivot = df.pivot_table(index="Date", columns="Symbol", values="Close")
    correlation = pivot.corr()
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(correlation, cmap="RdYlGn", center=0, linewidths=0.3)
    plt.title("Stock Price Correlation Matrix")
    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png", dpi=150)
    plt.close()
    print("Saved: outputs/correlation_heatmap.png")
    
    return pivot  # return pivot table for later use


# ---- MAIN: Run everything ----
if __name__ == "__main__":
    print("=== PHASE 1: Loading Data ===")
    raw_df = load_all_stocks(DATA_FOLDER)
    clean_df = clean_data(raw_df)
    pivot_df = run_eda(clean_df)
    
    # Save cleaned data for next phases
    clean_df.to_csv("data/cleaned_data.csv", index=False)
    print("\nPhase 1 complete! Saved: data/cleaned_data.csv")
