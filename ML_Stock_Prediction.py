import os
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import streamlit as st 

def fetch_sp500_data(filename="sp500.csv"):
    """Fetch S&P 500 data from Yahoo Finance or load from a local file."""
    if os.path.exists(filename):
        return pd.read_csv(filename, index_col=0, parse_dates=True)
    else:
        sp500 = yf.Ticker("^GSPC").history(period="max")
        sp500.to_csv(filename)
        return sp500

def preprocess_data(sp500):
    """Preprocess the S&P 500 data for analysis."""
    # Convert index to datetime and remove unnecessary columns
    sp500.index = pd.to_datetime(sp500.index)
    sp500.drop(columns=["Dividends", "Stock Splits"], inplace=True)
    
    # Add features for modeling
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    sp500 = sp500.loc["1990-01-01":].copy()
    return sp500

def add_features(sp500, horizons=[2,5,60,250,1000]):
    """Add rolling average and trend features to the DataFrame."""
    for horizon in horizons:
        rolling_averages = sp500.rolling(horizon).mean()
        sp500[f"Close_Ratio_{horizon}"] = sp500["Close"] / rolling_averages["Close"]
        sp500[f"Trend_{horizon}"] = sp500.shift(1).rolling(horizon).sum()["Target"]
    return sp500

def train_test_split(sp500, test_size=100):
    """Split the data into train and test sets."""
    return sp500.iloc[:-test_size], sp500.iloc[-test_size:]

def train_model(train, predictors):
    """Train the RandomForest model."""
    model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
    model.fit(train[predictors], train["Target"])
    return model

def evaluate_model(model, test, predictors):
    """Evaluate the model's performance."""
    preds = model.predict(test[predictors])
    precision = precision_score(test["Target"], preds)
    print(f"Precision Score: {precision}")
    return preds

def backtest(sp500, model, predictors, start=2500, step=250):
    """Perform backtesting with rolling prediction windows."""
    all_predictions = []
    for i in range(start, sp500.shape[0], step):
        train = sp500.iloc[0:i].copy()
        test = sp500.iloc[i:(i+step)].copy()
        preds = model.predict(test[predictors])
        all_predictions.append(pd.Series(preds, index=test.index, name="Predictions"))
    return pd.concat(all_predictions)

# Main function to orchestrate the data fetching, preprocessing, and modeling
def main():
    sp500 = fetch_sp500_data()
    sp500 = preprocess_data(sp500)
    sp500 = add_features(sp500)
    sp500.dropna(subset=sp500.columns[sp500.columns != "Tomorrow"], inplace=True)
    
    predictors = ["Close", "Volume", "Open", "High", "Low"] + [f"Close_Ratio_{h}" for h in [2,5,60,250,1000]] + [f"Trend_{h}" for h in [2,5,60,250,1000]]
    train, test = train_test_split(sp500)
    model = train_model(train, predictors)
    evaluate_model(model, test, predictors)

if __name__ == "__main__":
    main()
