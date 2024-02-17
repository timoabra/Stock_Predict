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
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[df.index.notna()]
    if not df.index.tz:
        df.index = df.index.tz_localize('UTC')
    else:
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
    preds = (preds >= 0.6).astype(int)  # Convert probabilities to 0 or 1
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def main():
    st.title("S&P 500 Stock Prediction")
    
    st.markdown("""
        ## Welcome to the S&P 500 Stock Predictor!
        This application leverages historical data to predict future movements of the S&P 500 index. Explore the app to view data insights and make predictions.
    """)

    # Adding an image to the application
    # Replace 'https://example.com/your_stock_image.jpg' with the actual path to your stock ticker photo or a valid URL
    st.image('https://example.com/your_stock_image.jpg', caption='S&P 500 Stock Movement Visualization')

    sp500 = load_data()
    prepared_sp500 = prepare_data(sp500)

    if prepared_sp500 is not None:
        if st.button("Show SP500 Data"):
            st.write(prepared_sp500.tail(5))

        if st.button("Plot Closing Prices by Year"):
            yearly_data = prepared_sp500['Close'].resample('Y').mean()  # Resample by year and take the mean
            st.line_chart(yearly_data)

        model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
        predictors = ["Close", "Volume", "Open", "High", "Low"]
        train = prepared_sp500.iloc[:-100]
        test = prepared_sp500.iloc[-100:]

        if st.button("Predict"):
            preds = predict(train, test, predictors, model)
            st.write(preds)

if __name__ == "__main__":
    main()
