# NIFTY-50 AI Investment Intelligence Platform 🚀

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nifty50-ai-platform-fdy6vzybd88nzsxi2g6xjj.streamlit.app/)

## 🌐 Live Application Demo
**The working prototype is fully deployed and available here:**
👉 **[Launch NIFTY-50 AI Platform](https://nifty50-ai-platform-fdy6vzybd88nzsxi2g6xjj.streamlit.app/)** 

## 📄 Technical Report (Deliverable 2)
The comprehensive technical report covering our methodology, exploratory data analysis, feature engineering, model architecture, risk assessment, explainability techniques, and portfolio logic is included in this repository.

👉 **[View Technical Report PDF](NIFTY50_Technical_Report.pdf)**

---

## 📊 Project Overview
An advanced, data-driven investment intelligence platform built using historical NIFTY-50 stock market data (Jan 2000 to April 2021). The platform transcends simple price forecasting by providing actionable decision-support tools, including interactive visual analytics, explainable AI predictions, automated risk assessment, and personalized portfolio construction.

### Features
- **Stock Price Prediction**: Uses XGBoost Ensembles to predict directional movement and exact closing prices.
- **Explainable AI (XAI)**: SHAP integration to provide transparent, indicator-level explanations for AI predictions.
- **Risk Assessment**: Classifies equities based on Sharpe Ratio, Volatility, and Maximum Drawdown.
- **Portfolio Construction**: Dynamically builds Conservative, Balanced, and Aggressive portfolios.
- **Market Anomaly Detection**: Statistical Z-score flagging of volatility regimes and volume spikes.

---

## 💻 Local Setup Instructions

*⚠️ Note: Due to GitHub's 100MB file size limits, the large compiled datasets (`featured_data.csv`, `cleaned_data.csv`) are **NOT** included in this repository. To run this project locally and verify the code, you must download the dataset separately.*

### 1. Download the Dataset
1. Go to: [Kaggle: NIFTY-50 Stock Market Data](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data)
2. Download the ZIP file, extract it, and place all the individual stock CSV files inside a folder named `data/` in the root directory.

### 2. Clone and Install
```bash
git clone https://github.com/Anningivivek/nifty50-ai-platform.git
cd nifty50-ai-platform

# Install required Python packages
pip install -r requirements.txt
```

### 3. Run the Data Pipeline
Execute the main pipeline script to parse the raw data, perform feature engineering, assess risk, train the XGBoost models, and generate the required pickle files:
```bash
python notebooks/nifty50_pipeline.py
```

### 4. Launch the Local Dashboard
Once the models are trained and saved, launch the interactive UI:
```bash
streamlit run src/phase5_dashboard.py
```

---

## 🛠️ Tech Stack
- **Machine Learning**: XGBoost, Scikit-learn, SHAP
- **Data Engineering**: Python, Pandas, NumPy
- **Frontend / Deployment**: Streamlit
