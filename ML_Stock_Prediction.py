import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

def load_data():
    today = datetime.now()
    future_date = today + timedelta(days=1)
    sp500 = yf.Ticker("^GSPC").history(start="1990-01-01", end=future_date.strftime("%Y-%m-%d"))
    sp500 = ensure_datetime_index_and_timezone(sp500)
    return sp500

def ensure_datetime_index_and_timezone(df):
    df.index = pd.to_datetime(df.index, errors='coerce').tz_localize('UTC')
    return df

def prepare_data(sp500):
    # Feature Engineering: Create lag features that are indicative of market trends
    sp500["Prev Close"] = sp500["Close"].shift(1)
    sp500["Volume Change"] = sp500["Volume"].pct_change()
    sp500["Daily Return"] = sp500["Close"].pct_change()
    # Create a target variable based on future movement (will not use in real-time prediction)
    sp500["Target"] = sp500["Daily Return"].shift(-1).apply(lambda x: 1 if x > 0 else 0)
    sp500 = sp500.dropna()
    return sp500

def train_test_split(sp500, split_date):
    train = sp500.loc[sp500.index < split_date]
    test = sp500.loc[sp500.index >= split_date]
    return train, test

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds = (preds >= 0.6).astype(int)
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def main():
    st.title("S&P 500 Stock Prediction")
    sp500 = load_data()
    sp500 = ensure_datetime_index_and_timezone(sp500)
    prepared_sp500 = prepare_data(sp500)

    # Define predictors excluding any future data
    predictors = ["Prev Close", "Volume Change", "Daily Return"]

    # Split data into training and testing sets
    split_date = datetime.now() - timedelta(days=365)  # Use last year for testing
    train, test = train_test_split(prepared_sp500, split_date.strftime("%Y-%m-%d"))

    model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

    # Example usage
    if st.button("Predict Future Movements"):
        preds = predict(train, test, predictors, model)
        st.write(preds)

if __name__ == "__main__":
    main()
