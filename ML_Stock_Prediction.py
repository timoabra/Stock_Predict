import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import pyzt

def load_data():
    # Fetch the latest data directly from Yahoo Finance
    sp500 = yf.Ticker("^GSPC").history(period="max")
    sp500 = ensure_datetime_index_and_timezone(sp500)
    return sp500

def ensure_datetime_index_and_timezone(df):
    # Convert index to datetime with UTC timezone
    df.index = pd.to_datetime(df.index, errors='coerce').tz_localize('UTC')
    st.write(datetime(2020, 1, 10, 10, 30, tzinfo=pytz.timezone("EST")))
    df = df[df.index.notna()]
    return df

def prepare_data(sp500):
    # Add instructions and prepare the data for analysis
    st.markdown("### Data Preparation Section")
    st.markdown("This section prepares the S&P 500 data for analysis, setting up the target variable and predictors.")
    
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    sp500 = sp500.drop(columns=["Dividends", "Stock Splits"]).dropna()
    sp500 = sp500.loc["1990-01-01":].copy()
    return sp500

def predict(train, test, predictors, model):
    # Predict future movements with a Random Forest model
    st.markdown("### Prediction Section")
    st.markdown("This section uses a Random Forest model to predict future movements of the S&P 500 index based on historical data.")
    
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
        This application leverages historical data to predict future movements of the S&P 500 index. 
        Use the buttons below to explore the app:
    """)

    # Adding the image back into the application
    st.image('https://media.istockphoto.com/id/523155194/photo/financial-data-on-a-monitor-stock-market-data-on-led.jpg?s=612x612&w=0&k=20&c=_3Rm4QHvucRzhrosmVUPUQoDx8h-E35DijJsbtQS5mY=', 
             caption='S&P 500 Stock Movement Visualization', use_column_width=True)

    sp500 = load_data()
    prepared_sp500 = prepare_data(sp500)

    if prepared_sp500 is not None:
        if st.button("Show Latest SP500 Data"):
            st.write(prepared_sp500.tail(5))

        if st.button("Plot Annual Closing Prices"):
            st.markdown("### Annual Closing Prices")
            annual_data = prepared_sp500['Close'].resample('Y').mean()
            st.line_chart(annual_data, use_container_width=True)

        model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
        predictors = ["Close", "Volume", "Open", "High", "Low"]
        train = prepared_sp500.iloc[:-100]
        test = prepared_sp500.iloc[-100:]

        if st.button("Predict Future Movements"):
            preds = predict(train, test, predictors, model)
            st.write("Predictions for the next 100 days:")
            st.write(preds)

if __name__ == "__main__":
    main()

