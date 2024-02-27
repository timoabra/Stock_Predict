import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

def load_data():
    """
    Fetches the latest available S&P 500 index data from Yahoo Finance, starting from January 1, 1990, to ensure a comprehensive dataset for analysis. This function dynamically adjusts the end date to one day in the future to include the most recent complete trading data.
    """
    today = datetime.now()
    future_date = today + timedelta(days=1)  # Ensures inclusion of the last available data
    sp500 = yf.Ticker("^GSPC").history(start="1990-01-01", end=future_date.strftime("%Y-%m-%d"))
    sp500 = ensure_datetime_index_and_timezone(sp500)
    return sp500

def ensure_datetime_index_and_timezone(df):
    """
    Adjusts the DataFrame's datetime index to be timezone-aware, standardizing it to UTC. This adjustment is crucial for consistent analysis, especially when dealing with time series data that spans multiple time zones.
    """
    if df.index.tz is None:  # Check if the index is naive
        df.index = pd.to_datetime(df.index, errors='coerce').tz_localize('UTC')
    else:  # If already timezone-aware, convert to UTC
        df.index = df.index.tz_convert('UTC')
    return df

def prepare_data(sp500):
    """
    Prepares the S&P 500 data for subsequent analysis by:
    - Creating a 'Tomorrow' column to hold the next day's closing prices, enabling the calculation of the binary 'Target' variable.
    - The 'Target' variable indicates whether the market will go up (1) or down (0) the next day, based on the closing price.
    - Cleans the dataset by dropping unnecessary columns and rows with missing values, focusing the analysis on clean, relevant data.
    """
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    sp500 = sp500.drop(columns=["Dividends", "Stock Splits"]).dropna()
    sp500 = sp500.loc["1990-01-01":].copy()
    return sp500

def predict(train, test, predictors, model):
    """
    Employs a RandomForestClassifier model to predict the market's direction on the following day:
    - The model is trained using historical data, focusing on whether the closing price will rise (1) or fall (0).
    - Predictions are based on a probability threshold of 60%; if the model's confidence exceeds this, it predicts an increase (1), otherwise a decrease (0).
    - This function returns a DataFrame combining actual outcomes and the model's predictions, facilitating performance evaluation.
    """
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds = (preds >= 0.6).astype(int)  # Threshold for predicting market rise
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def main():
    st.title("S&P 500 Stock Prediction")
    
    # Other parts of your main function remain unchanged...

if __name__ == "__main__":
    main()
