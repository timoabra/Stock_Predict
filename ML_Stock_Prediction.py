import streamlit as st
import pandas as pd
import os
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Function to load or fetch data
@st.cache
def get_data():
    if "sp500.csv" not in os.listdir():
        sp500 = yf.Ticker("^GSPC").history(period="max")
        sp500.to_csv("sp500.csv")
    else:
        sp500 = pd.read_csv("sp500.csv", index_col='Date', parse_dates=True)
    return sp500

# Data preparation and feature engineering
@st.cache
def prepare_data(sp500):
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    sp500.dropna(inplace=True)
    horizons = [2, 5, 60, 250, 1000]
    for horizon in horizons:
        sp500[f"Close_Ratio_{horizon}"] = sp500["Close"] / sp500["Close"].rolling(window=horizon).mean()
        sp500[f"Trend_{horizon}"] = sp500["Target"].rolling(window=horizon).sum()
    sp500.dropna(inplace=True)
    predictors = ["Close", "Volume", "Open", "High", "Low"] + [f"Close_Ratio_{horizon}" for horizon in horizons] + [f"Trend_{horizon}" for horizon in horizons]
    return sp500, predictors

# Train model
@st.cache(allow_output_mutation=True)
def train_model(data, predictors):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    model.fit(train_data[predictors], train_data["Target"])
    return model, test_data

# Displaying results
def display_results(model, test_data, predictors):
    test_predictions = model.predict(test_data[predictors])
    precision = precision_score(test_data["Target"], test_predictions)
    recall = recall_score(test_data["Target"], test_predictions)
    f1 = f1_score(test_data["Target"], test_predictions)
    roc_auc = roc_auc_score(test_data["Target"], model.predict_proba(test_data[predictors])[:, 1])

    st.metric("Precision", precision)
    st.metric("Recall", recall)
    st.metric("F1 Score", f1)
    st.metric("ROC AUC", roc_auc)

# Streamlit UI
def main():
    st.title("S&P 500 Prediction Model")

    if st.button("Load Data and Train Model"):
        data = get_data()
        prepared_data, predictors = prepare_data(data)
        model, test_data = train_model(prepared_data, predictors)
        display_results(model, test_data, predictors)

if __name__ == "__main__":
    main()





