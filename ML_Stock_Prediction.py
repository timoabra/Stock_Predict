import streamlit as st
import yfinance as yf
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Function to load or fetch the data
def load_data(ticker_symbol):
    if os.path.exists(f"{ticker_symbol}.csv"):
        data = pd.read_csv(f"{ticker_symbol}.csv", index_col=0, parse_dates=True)
    else:
        data = yf.Ticker(ticker_symbol).history(period="max")
        data.to_csv(f"{ticker_symbol}.csv")
    return data

# Function to explicitly convert index to DateTimeIndex and adjust timezone
def ensure_datetime_index_and_timezone(df):
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[~df.index.isna()]
    if df.index.tz is not None:
        df.index = df.index.tz_convert('UTC')
    else:
        df.index = df.index.tz_localize('UTC')
    return df

# Prepare the data for modeling
def prepare_data(data):
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    data = data.drop(columns=["Dividends", "Stock Splits"]).dropna()
    data = data.loc["1990-01-01":].copy()
    return data

# Prediction function
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= 0.6] = 1
    preds[preds < 0.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Main Streamlit app function
def main():
    st.title("S&P 500 Stock Prediction")

    # List of S&P 500 ticker symbols
    sp500_symbols = [...]  # Replace with your list of symbols

    # Create a dropdown menu
    selected_symbol = st.selectbox('Select a stock:', sp500_symbols)

    # Fetch the data for the selected stock
    data = load_data(selected_symbol)
    data = ensure_datetime_index_and_timezone(data)
    prepared_data = prepare_data(data)

    if prepared_data is not None:
        # Adjust here to show the most recent data in the DataFrame
        recent_data_length = 5  # Number of most recent rows to display
        if st.button("Show Data"):
            st.write(prepared_data.tail(recent_data_length))

        if st.button("Plot Closing Prices"):
            st.line_chart(prepared_data['Close'])

        model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
        predictors = ["Close", "Volume", "Open", "High", "Low"]
        train = prepared_data.iloc[:-100]
        test = prepared_data.iloc[-100:]

        if st.button("Predict"):
            preds = predict(train, test, predictors, model)
            precision = precision_score(test["Target"], preds["Predictions"], zero_division=0)
            st.write(f"Precision: {precision}")
            st.write(preds)

if __name__ == "__main__":
    main()
