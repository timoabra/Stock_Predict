import streamlit as st
import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

def load_data():
    if os.path.exists("sp500.csv"):
        sp500 = pd.read_csv("sp500.csv", index_col=0, parse_dates=True)
    else:
        sp500 = yf.Ticker("^GSPC").history(period="max")
        sp500.to_csv("sp500.csv")
    return sp500

def ensure_datetime_index_and_timezone(df):
    # Ensure the index is a DatetimeIndex
    df.index = pd.to_datetime(df.index, errors='coerce')

    # Filter out rows where the index is NaT
    df = df[df.index.notna()]

    # Robustly check and localize/convert timezone only if necessary
    if df.index.tzinfo is None:
        # Localize the index to UTC if it doesn't have timezone information
        df.index = df.index.tz_localize('UTC')
    elif str(df.index.tzinfo) != 'UTC':
        # Convert to UTC if it has timezone information that is not UTC
        df.index = df.index.tz_convert('UTC')

    return df

def prepare_data(sp500):
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    sp500 = sp500.drop(columns=["Dividends", "Stock Splits"]).dropna()
    sp500 = sp500.loc["1990-01-01":].copy()
    return sp500

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= 0.6] = 1
    preds[preds < 0.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def main():
    st.title("S&P 500 Stock Prediction")
    
    st.markdown("""
        ## Welcome to the S&P 500 Stock Predictor!
        This application leverages historical data to predict future movements of the S&P 500 index. Explore the app to view data insights and make predictions.
        
        **How to use:**
        - **Show SP500 Data:** Displays the most recent data fetched for the S&P 500.
        - **Plot Closing Prices:** Visualizes the S&P 500 closing prices over time.
        - **Predict:** Uses a machine learning model to predict future price movements.
    """)

    sp500 = load_data()
    sp500 = ensure_datetime_index_and_timezone(sp500)
    prepared_sp500 = prepare_data(sp500)

    if prepared_sp500 is not None:
        if st.button("Show SP500 Data"):
            st.write(prepared_sp500.tail(5))

        if st.button("Plot Closing Prices"):
            st.line_chart(prepared_sp500['Close'])

        model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
        predictors = ["Close", "Volume", "Open", "High", "Low"]
        train = prepared_sp500.iloc[:-100]
        test = prepared_sp500.iloc[-100:]

        if st.button("Predict"):
            preds = predict(train, test, predictors, model)
            precision = precision_score(test["Target"], preds["Predictions"], zero_division=0)
            st.write(f"Precision: {precision}")
            st.write(preds)

if __name__ == "__main__":
    main()

