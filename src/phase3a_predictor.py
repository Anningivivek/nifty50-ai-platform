# ============================================================
# PHASE 3A: STOCK PREDICTOR ENGINE
# ============================================================
# We train two models:
# Model 1: XGBoost (fast, good with tabular data)
# Model 2: LSTM (deep learning, good with time series)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ---- MODEL 1: XGBoost Direction Classifier ----

def train_xgboost(df, symbol="RELIANCE"):
    """
    Trains XGBoost to predict: will price go UP tomorrow?
    Features: RSI, MACD, Bollinger, Moving Averages, Volatility
    Target: 1 (up) or 0 (down)
    """
    print(f"\nTraining XGBoost for: {symbol}")
    
    # Filter to one stock
    stock_df = df[df["Symbol"] == symbol].copy()
    
    # Features we feed into the model
    feature_cols = [
        "RSI", "MACD", "MACD_Signal", "MACD_Hist",
        "MA_20", "MA_50", "BB_Width", "Volatility_20",
        "Daily_Return", "Volume"
    ]
    
    # Remove any missing values
    stock_df = stock_df.dropna(subset=feature_cols + ["Target_Direction"])
    
    X = stock_df[feature_cols].values
    y = stock_df["Target_Direction"].values
    
    # Split: use 80% for training, 20% for testing
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train XGBoost
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=1,  # Prevent macOS threading deadlock
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Directional Accuracy: {accuracy:.2%}")
    
    # Save model
    joblib.dump(model, f"models/xgboost_{symbol}.pkl")
    print(f"Model saved: models/xgboost_{symbol}.pkl")
    
    return model, accuracy

def train_xgboost_regressor(df, symbol="RELIANCE"):
    """
    Trains XGBoost to predict the exact Next_Close price.
    Features: RSI, MACD, Bollinger, Moving Averages, Volatility
    Target: Next_Close (continuous)
    """
    # Filter to one stock
    stock_df = df[df["Symbol"] == symbol].copy()
    
    # Features we feed into the model
    feature_cols = [
        "RSI", "MACD", "MACD_Signal", "MACD_Hist",
        "MA_20", "MA_50", "BB_Width", "Volatility_20",
        "Daily_Return", "Volume"
    ]
    
    # Remove any missing values
    stock_df = stock_df.dropna(subset=feature_cols + ["Next_Close"])
    
    X = stock_df[feature_cols].values
    y = stock_df["Next_Close"].values
    
    # Split: use 80% for training, 20% for testing
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train XGBoost Regressor
    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        n_jobs=1,  # Prevent macOS threading deadlock
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Save model
    joblib.dump(model, f"models/xgboost_reg_{symbol}.pkl")
    
    return model, mae, rmse


# ---- MODEL 2: LSTM Price Forecaster ----

def prepare_lstm_data(series, lookback=60):
    """
    LSTM needs sequences. 
    For each day, we look at the last 60 days of prices to predict tomorrow.
    Input shape: (samples, 60, 1)
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])
    
    return np.array(X), np.array(y), scaler


def train_lstm(df, symbol="RELIANCE"):
    """
    Trains an LSTM neural network to predict next day's closing price.
    """
    print(f"\nTraining LSTM for: {symbol}")
    
    stock_df = df[df["Symbol"] == symbol].sort_values("Date")
    prices = stock_df["Close"].values
    
    LOOKBACK = 60  # use last 60 days to predict next day
    
    X, y, scaler = prepare_lstm_data(prices, lookback=LOOKBACK)
    
    # Reshape for LSTM: (samples, timesteps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # 80/20 train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)  # output: predicted price (scaled)
    ])
    
    model.compile(optimizer="adam", loss="mean_squared_error")
    
    # Train (stop early if no improvement)
    early_stop = EarlyStopping(monitor="val_loss", patience=5)
    model.fit(
        X_train, y_train,
        epochs=5,  # Reduced from 30 for faster demonstration
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1  # Show progress bar!
    )
    
    # Evaluate
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    print(f"MAE: {mae:.2f}  |  RMSE: {rmse:.2f}")
    
    # Plot prediction vs actual
    plt.figure(figsize=(12, 5))
    plt.plot(y_actual, label="Actual Price", linewidth=1.5)
    plt.plot(y_pred, label="Predicted Price", linewidth=1.5, linestyle="--")
    plt.title(f"LSTM Price Prediction - {symbol}")
    plt.xlabel("Days")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/lstm_prediction_{symbol}.png", dpi=150)
    plt.close()
    
    # Save model
    model.save(f"models/lstm_{symbol}.h5")
    print(f"Model saved: models/lstm_{symbol}.h5")
    
    return model, mae, rmse


# ---- MAIN ----
if __name__ == "__main__":
    print("=== PHASE 3A: Stock Predictor ===")
    
    df = pd.read_csv("data/featured_data.csv", parse_dates=["Date"])
    
    # Train for all stocks!
    target_stocks = df["Symbol"].unique()
    
    results = {}
    for sym in target_stocks:
        if sym in df["Symbol"].values:
            xgb_model, acc = train_xgboost(df, symbol=sym)
            xgb_reg, mae, rmse = train_xgboost_regressor(df, symbol=sym)
            results[sym] = {"Accuracy": acc, "MAE": mae, "RMSE": rmse}
    
    print("\n=== RESULTS SUMMARY ===")
    for sym, metrics in results.items():
        print(f"{sym}: Accuracy={metrics['Accuracy']:.2%}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")
    
    print("\nPhase 3A complete!")
