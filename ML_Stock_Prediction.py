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
    today = datetime.now().date()

    if os.path.exists(file_path):
        sp500 = pd.read_csv(file_path, index_col=0, parse_dates=True)
        sp500.index = pd.to_datetime(sp500.index)
        last_date = sp500.index.max().date()

        if last_date < today:
            start_date = last_date + pd.Timedelta(days=1)
            new_data = yf.download("^GSPC", start=start_date.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))
            if not new_data.empty:
                new_data.to_csv(file_path, mode='a', header=False)
                sp500 = pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        sp500 = yf.download("^GSPC", period="max")
        sp500.to_csv(file_path)
    
    return sp500

def ensure_datetime_index_and_timezone(df):
    df.index = pd.to_datetime(df.index, errors='coerce').tz_localize(None)
    df = df[~df.index.isna()]
    return df

def prepare_data(sp500):
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    sp500 = sp500.dropna()
    sp500 = sp500.loc["1990-01-01":].copy()
    return sp500

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
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
        precision = precision_score(test["Target"], preds["Predictions"])
        st.write(f"Precision: {precision}")
        st.write(preds)

if __name__ == "__main__":
    main()
