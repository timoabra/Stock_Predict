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
    df.index = pd.to_datetime(df.index, errors='coerce')
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
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
    
    st.markdown("""
        ## Welcome to the S&P 500 Stock Predictor!
        This application leverages historical data to predict future movements of the S&P 500 index, using a machine learning model to forecast whether the index will rise or fall the next day.
    """)

    # Visual representation of stock movements
    st.image('https://media.istockphoto.com/id/523155194/photo/financial-data-on-a-monitor-stock-market-data-on-led.jpg?s=612x612&w=0&k=20&c=_3Rm4QHvucRzhrosmVUPUQoDx8h-E35DijJsbtQS5mY=', 
             caption='S&P 500 Stock Movement Visualization', use_column_width=True)

    sp500 = load_data()
    sp500 = ensure_datetime_index_and_timezone(sp500)
    prepared_sp500 = prepare_data(sp500)

    if prepared_sp500 is not None:
        if st.button("Show Latest SP500 Data"):
            st.write(prepared_sp500.tail(5))

        if st.button("Plot Annual Closing Prices"):
            # This graph visualizes the S&P 500's annual closing prices, offering insights into its long-term growth trends over time.
            st.markdown("### Annual Closing Prices Visualization")
            st.markdown("This graph displays the S&P 500's annual closing prices, highlighting the index's growth and fluctuations over an extended period. It's a powerful way to observe macroeconomic trends and market behavior year over year.")
            annual_data = prepared_sp500['Close'].resample('Y').mean()
            st.line_chart(annual_data, use_container_width=True)

        model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
        predictors = ["Close", "Volume", "Open", "High", "Low"]
        train = prepared_sp500.iloc[:-100]
        test = prepared_sp500.iloc[-100:]

        if st.button("Predict Future Movements"):
            # The prediction functionality offers insights into potential future market movements, aiding in investment and financial planning.
            preds = predict(train, test, predictors, model)
            st.markdown("### Prediction of Future Market Movements")
            st.markdown("This section predicts whether the S&P 500 index will experience an upward (1) or downward (0) movement the following day. A prediction of '1' suggests a rise with more than 60% confidence, while '0' indicates a lower confidence in upward movement.")
            st.write("Shows historical prediction as well as predictions to come:"
            st.write(preds)

if __name__ == "__main__":
    main()
