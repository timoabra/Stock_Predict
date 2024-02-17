import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

def load_data():
    """
    Fetches the latest available S&P 500 index data from Yahoo Finance.
    The data range starts from January 1, 1990, to ensure a comprehensive dataset,
    extending to the most recent trading day to keep the analysis current.
    """
    today = datetime.now()
    future_date = today + timedelta(days=1)  # Ensures inclusion of the last available data
    sp500 = yf.Ticker("^GSPC").history(start="1990-01-01", end=future_date.strftime("%Y-%m-%d"))
    sp500 = ensure_datetime_index_and_timezone(sp500)
    return sp500

def ensure_datetime_index_and_timezone(df):
    """
    Ensures that the DataFrame's datetime index is timezone-aware and standardized to UTC.
    This is crucial for consistent time series analysis, especially when merging data
    from different sources that may have varying timezone settings.
    """
    df.index = pd.to_datetime(df.index, errors='coerce')
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    return df

def prepare_data(sp500):
    """
    Prepares the S&P 500 dataset for analysis and prediction by:
    - Shifting the 'Close' prices to create a 'Tomorrow' column, which is used to calculate the target variable.
    - Creating a binary 'Target' variable to indicate whether the market goes up (1) or down (0) the next day.
    - Dropping unnecessary columns and any rows with missing values to clean the dataset.
    - Slicing the dataset to focus on data from 1990 onwards for a more relevant analysis period.
    """
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    sp500 = sp500.drop(columns=["Dividends", "Stock Splits"]).dropna()
    sp500 = sp500.loc["1990-01-01":].copy()
    return sp500

def predict(train, test, predictors, model):
    """
    Trains a RandomForestClassifier model on the training dataset and makes predictions on the test dataset.
    This function:
    - Fits the model using the specified predictors and the binary 'Target' as the output variable.
    - Predicts the probability of the market going up the next day and converts these probabilities to binary predictions.
    - Returns a DataFrame combining the actual and predicted values for comparison.
    """
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds = (preds >= 0.6).astype(int)  # Threshold of 0.6 to decide between 0 and 1
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def main():
    """
    The main function orchestrates the Streamlit application, including:
    - Displaying the application title and an introductory markdown section.
    - Visualizing the S&P 500 stock movement with an image.
    - Loading, preparing, and displaying the latest S&P 500 data.
    - Providing buttons to display the latest data, plot annual closing prices, and predict future market movements.
    """
    st.title("S&P 500 Stock Prediction")
    
    st.markdown("""
        ## Welcome to the S&P 500 Stock Predictor!
        This application leverages historical data to predict future movements of the S&P 500 index. 
        Use the buttons below to explore the app:
    """)

    # Display a relevant image for visual engagement
    st.image('https://media.istockphoto.com/id/523155194/photo/financial-data-on-a-monitor-stock-market-data-on-led.jpg?s=612x612&w=0&k=20&c=_3Rm4QHvucRzhrosmVUPUQoDx8h-E35DijJsbtQS5mY=', 
             caption='S&P 500 Stock Movement Visualization', use_column_width=True)

    sp500 = load_data()
    prepared_sp500 = prepare_data(sp500)

    if prepared_sp500 is not None:
        if st.button("Show Latest SP500 Data"):
            st.write(prepared_sp500.tail(5))

        if st.button("Plot Annual Closing Prices"):
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
