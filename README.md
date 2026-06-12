# NIFTY-50 AI Investment Intelligence Platform

## Project Overview
An AI-powered investment intelligence platform built for the NIFTY-50 dataset
covering Jan 2000 to April 2021.

## Features
- Stock price prediction using LSTM & XGBoost
- Risk assessment (Sharpe Ratio, Sortino Ratio, Max Drawdown)
- Portfolio construction for 3 investor profiles
- Market anomaly detection
- Interactive Streamlit dashboard

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/nifty50-intelligence.git
cd nifty50-intelligence
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Go to: https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data
Download all CSV files and place them in the `data/` folder.

### 4. Run the pipeline in order
```bash
python src/phase1_load_data.py
python src/phase2_features.py
python src/phase3a_predictor.py
python src/phase3b_risk.py
python src/phase4a_portfolio.py
python src/phase4b_anomaly.py
```

### 5. Launch the Dashboard
```bash
streamlit run src/phase5_dashboard.py
```

## Evaluation Metrics Used
- **Direction Prediction**: Accuracy, F1 Score
- **Price Forecasting**: MAE, RMSE, R²
- **Risk**: Sharpe Ratio, Sortino Ratio, Max Drawdown, VaR

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow/Keras, Streamlit, Plotly
