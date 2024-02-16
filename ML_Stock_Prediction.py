import streamlit as st
import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Updated load_data function
def load_data():
    file_path = "sp500.csv"
    if os.path.exists(file_path):
        sp500 = pd.read_csv(file_path, index_col=0, parse_dates=True)
        last_date = sp500.index[-1]
        today = pd.Timestamp(datetime.today().strftime('%Y-%m-%d'))
        if last_date < today:
            new_data = yf.Ticker("^GSPC").history(start=last_date, end=today)
            if not new_data.empty:
                new_data = new_data.loc[:, sp500.columns]
                sp500 = pd.concat([sp500, new_data])
                sp500 = sp500[~sp500.index.duplicated(keep='first')]
                sp500.to_csv(file_path)
    else:
        sp500 = yf.Ticker("^GSPC").history(period="max")
        sp500.to_csv(file_path)
    return sp500

# Function to explicitly convert index to DateTimeIndex and adjust timezone
def ensure_datetime_index_and_timezone(df):
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[~df.index.isna()]
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    return df

# Prepare the data for modeling
def prepare_data(sp500):
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    sp500 = sp500.dropna()
    sp500 = sp500.loc["1990-01-01":].copy()
    return sp500

# Prediction function
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Main Streamlit app function
def main():
    st.title("S&P 500 Stock Prediction")
    
    # Description and Instructions using Markdown
    st.markdown("""
        ## Welcome to the S&P 500 Stock Predictor!
        This app predicts the movement of the S&P 500 stock prices. Use the buttons below to view the latest S&P 500 data, visualize closing prices, and predict future movements based on historical data.
        
        **Instructions:**
        - Click on *Show SP500 Data* to view the most recent stock data.
        - Click on *Plot Closing Prices* to see a chart of recent closing prices.
        - Click on *Predict* to view our model's predictions for the stock's movement.
    """)
    
    # Adding a stock ticker image
    st.image('stock_photo.jpg', caption='S&P 500 Stock Movement')
    
    sp500 = load_data()
    sp500 = ensure_datetime_index_and_timezone(sp500)
    prepared_sp500 = prepare_data(sp500)
    
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
