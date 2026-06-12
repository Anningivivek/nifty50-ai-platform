# ============================================================
# PHASE 5: STREAMLIT INTERACTIVE DASHBOARD
# ============================================================
# Run this with: streamlit run src/phase5_dashboard.py
#
# This creates a web app with 4 tabs:
# 1. Stock Analysis — price charts + indicators
# 2. Stock Predictor — predictions from our ML models
# 3. Risk Assessment — Sharpe ratio, drawdown etc.
# 4. Portfolio Builder — 3 investor profiles

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os

# ---- Page Setup ----
st.set_page_config(
    page_title="NIFTY-50 Investment Intelligence",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NIFTY-50 AI Investment Intelligence Platform")
st.markdown("*Data-driven insights for smarter investing*")

# ---- Load Data ----
@st.cache_data  # Cache so it doesn't reload every time user clicks
def load_data():
    df = pd.read_csv("data/featured_data.zip", compression="zip", parse_dates=["Date"])
    risk_df = pd.read_csv("data/risk_profiles.csv")
    return df, risk_df

df, risk_df = load_data()
all_symbols = sorted(df["Symbol"].unique().tolist())

# ---- TABS ----
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Stock Analysis",
    "🤖 Stock Predictor", 
    "⚠️ Risk Assessment",
    "💼 Portfolio Builder"
])

# ===============================
# TAB 1: STOCK ANALYSIS
# ===============================
with tab1:
    st.header("Stock Price Analysis")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_stock = st.selectbox("Select Stock", all_symbols)
        show_ma = st.checkbox("Show Moving Averages", value=True)
        show_bollinger = st.checkbox("Show Bollinger Bands", value=False)
        date_range = st.selectbox("Time Range", ["1 Year", "3 Years", "5 Years", "All Time"])
    
    with col2:
        stock_data = df[df["Symbol"] == selected_stock].sort_values("Date")
        
        # Filter by date range
        if date_range == "1 Year":
            stock_data = stock_data.tail(252)
        elif date_range == "3 Years":
            stock_data = stock_data.tail(252 * 3)
        elif date_range == "5 Years":
            stock_data = stock_data.tail(252 * 5)
        
        # Create candlestick chart
        fig = go.Figure()
        
        # Main price line
        fig.add_trace(go.Scatter(
            x=stock_data["Date"], y=stock_data["Close"],
            name="Close Price", line=dict(color="royalblue", width=2)
        ))
        
        if show_ma:
            fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["MA_20"],
                                     name="MA 20", line=dict(color="orange", width=1.5, dash="dot")))
            fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["MA_50"],
                                     name="MA 50", line=dict(color="green", width=1.5, dash="dot")))
        
        if show_bollinger:
            fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["BB_Upper"],
                                     name="BB Upper", line=dict(color="red", dash="dash"), opacity=0.5))
            fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["BB_Lower"],
                                     name="BB Lower", line=dict(color="red", dash="dash"), opacity=0.5,
                                     fill="tonexty", fillcolor="rgba(255,0,0,0.05)"))
        
        fig.update_layout(title=f"{selected_stock} Price Chart",
                          xaxis_title="Date", yaxis_title="Price (₹)",
                          height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
        # RSI Chart
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["RSI"],
                                      name="RSI", line=dict(color="purple")))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig_rsi.update_layout(title="RSI Indicator", height=250, template="plotly_white")
        st.plotly_chart(fig_rsi, use_container_width=True)


# ===============================
# TAB 2: STOCK PREDICTOR
# ===============================
with tab2:
    st.header("AI Stock Predictor")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        pred_stock = st.selectbox("Select Stock to Predict", all_symbols, key="pred_stock")
        st.markdown("### Model Explanation")
        st.info("""
        **XGBoost Classifier** predicts whether the stock will go UP or DOWN tomorrow.
        
        **XGBoost Regressor** predicts the exact closing price for the next day.
        
        **Explainable AI (SHAP)** shows which financial indicators drove the prediction.
        """)
    
    with col2:
        # Load and use XGBoost models
        model_path = f"models/xgboost_{pred_stock}.pkl"
        reg_path = f"models/xgboost_reg_{pred_stock}.pkl"
        
        if os.path.exists(model_path) and os.path.exists(reg_path):
            model = joblib.load(model_path)
            reg_model = joblib.load(reg_path)
            
            # Get latest data for this stock
            latest = df[df["Symbol"] == pred_stock].sort_values("Date").tail(1)
            
            feature_cols = ["RSI", "MACD", "MACD_Signal", "MACD_Hist",
                            "MA_20", "MA_50", "BB_Width", "Volatility_20",
                            "Daily_Return", "Volume"]
            
            available_features = [f for f in feature_cols if f in latest.columns]
            X_latest = latest[available_features].values
            
            # Direction Prediction
            prediction = model.predict(X_latest)[0]
            probability = model.predict_proba(X_latest)[0]
            
            # Price Prediction
            predicted_price = reg_model.predict(X_latest)[0]
            current_price = latest["Close"].values[0]
            
            st.markdown("### Tomorrow's Prediction")
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                if prediction == 1:
                    st.success(f"📈 **UP**")
                else:
                    st.error(f"📉 **DOWN**")
            with mcol2:
                st.metric("Confidence", f"{probability[prediction]:.1%}")
            with mcol3:
                st.metric("Predicted Close", f"₹{predicted_price:.2f}", delta=f"{predicted_price - current_price:.2f}")
            
            st.caption("⚠️ This is a model prediction, not financial advice.")
            
            # SHAP Explainability
            st.markdown("### Why did the AI make this prediction?")
            import shap
            import matplotlib.pyplot as plt
            
            # TreeExplainer works natively with XGBoost
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(latest[available_features])
            
            # SHAP Waterfall plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("SHAP Waterfall Plot: Red bars push the prediction UP, blue bars push it DOWN.")
        else:
            st.warning(f"Model not trained yet for {pred_stock}. Run Phase 3A first.")


# ===============================
# TAB 3: RISK ASSESSMENT
# ===============================
with tab3:
    st.header("Risk Assessment Dashboard")
    
    # Risk overview table
    st.subheader("All Stocks Risk Profile")
    
    display_cols = ["Symbol", "Annual_Return", "Annual_Volatility", 
                    "Sharpe_Ratio", "Max_Drawdown", "Risk_Level"]
    
    # Color the risk levels
    styled_risk = risk_df[display_cols].copy()
    styled_risk["Annual_Return"] = styled_risk["Annual_Return"].map("{:.1%}".format)
    styled_risk["Annual_Volatility"] = styled_risk["Annual_Volatility"].map("{:.1%}".format)
    styled_risk["Max_Drawdown"] = styled_risk["Max_Drawdown"].map("{:.1%}".format)
    
    st.dataframe(styled_risk.sort_values("Sharpe_Ratio", ascending=False), 
                 use_container_width=True, height=300)
    
    # Risk-Return scatter plot
    st.subheader("Risk vs Return Chart")
    fig_scatter = px.scatter(
        risk_df,
        x="Annual_Volatility",
        y="Annual_Return",
        color="Risk_Level",
        hover_name="Symbol",
        color_discrete_map={"LOW RISK": "green", "MEDIUM RISK": "orange", "HIGH RISK": "red"},
        title="Risk vs Return: All NIFTY-50 Stocks",
        labels={"Annual_Volatility": "Annual Volatility (Risk)", 
                "Annual_Return": "Annual Return"},
        height=500
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ===============================
# TAB 4: PORTFOLIO BUILDER
# ===============================
with tab4:
    st.header("Portfolio Builder")
    
    investor_type = st.radio(
        "Select Your Investor Profile:",
        ["Conservative 🛡️", "Balanced ⚖️", "Aggressive 🚀"],
        horizontal=True
    )
    
    # Map to profile name
    profile_map = {
        "Conservative 🛡️": "Conservative",
        "Balanced ⚖️": "Balanced",
        "Aggressive 🚀": "Aggressive"
    }
    profile = profile_map[investor_type]
    
    # Show explanation
    descriptions = {
        "Conservative": "Focus on stable, low-risk stocks. Suitable for capital preservation. Expect moderate but steady returns.",
        "Balanced": "Mix of stable and growth stocks. Good for medium-term goals like buying a house or retirement planning.",
        "Aggressive": "High-growth potential stocks with higher risk. Suitable for long-term wealth building (5+ years horizon)."
    }
    st.info(descriptions[profile])
    
    # Build portfolio
    if profile == "Conservative":
        selected = risk_df[risk_df["Risk_Level"] == "LOW RISK"].nlargest(8, "Sharpe_Ratio")
    elif profile == "Balanced":
        low = risk_df[risk_df["Risk_Level"] == "LOW RISK"].nlargest(5, "Sharpe_Ratio")
        med = risk_df[risk_df["Risk_Level"] == "MEDIUM RISK"].nlargest(3, "Annual_Return")
        selected = pd.concat([low, med])
    else:
        selected = risk_df.nlargest(8, "Annual_Return")
    
    # Equal weight
    n = len(selected)
    if n == 0:
        st.warning(f"⚠️ No stocks met the strict criteria for the {profile} profile. Please select another profile like 'Balanced' or 'Aggressive'.")
    else:
        selected["Allocation"] = 1 / n
        selected["Amount_for_1L"] = selected["Allocation"] * 100000
        
        # Display
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Recommended Stocks")
            display = selected[["Symbol", "Allocation", "Sharpe_Ratio", "Annual_Return", "Risk_Level"]].copy()
            display["Allocation"] = display["Allocation"].map("{:.1%}".format)
            display["Annual_Return"] = display["Annual_Return"].map("{:.1%}".format)
            st.dataframe(display, use_container_width=True)
        
        with col2:
            st.subheader("Allocation Pie Chart")
            fig_pie = px.pie(
                selected,
                values="Allocation",
                names="Symbol",
                title=f"{profile} Portfolio Allocation"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    st.markdown("**⚠️ Disclaimer:** This platform is for educational purposes only. "
                "Do not use this as actual financial advice. Consult a SEBI-registered advisor.")
