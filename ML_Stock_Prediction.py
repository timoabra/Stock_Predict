import yfinance as yf
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Ensure datetime index and handle timezone appropriately
def ensure_datetime_index_and_timezone(df):
    # Check if the index is a DatetimeIndex and has a timezone
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)  # Remove timezone information
    else:
        df.index = pd.to_datetime(df.index)  # Convert to DatetimeIndex without timezone
        df.index = df.index.tz_localize(None)  # Ensure it is timezone naive
    
    # Convert any datetime columns to be timezone naive for consistency
    for col in df.select_dtypes(include=['datetime', 'datetime64']):
        df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)
    
    return df

# Function to load or fetch the data
def load_data():
    if os.path.exists("sp500.csv"):
        sp500 = pd.read_csv("sp500.csv", index_col=0, parse_dates=True)
        sp500 = ensure_datetime_index_and_timezone(sp500)
    else:
        sp500 = yf.Ticker("^GSPC").history(period="max")
        sp500.to_csv("sp500.csv")
    return sp500

# Function to prepare the data for modeling
def prepare_data(sp500):
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    sp500 = sp500.drop(columns=["Dividends", "Stock Splits"])
    sp500 = sp500.loc["1990-01-01":].copy()
    return sp500

# Function to predict
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds = (preds >= 0.5).astype(int)
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Backtest function
def backtest(data, model, predictors):
    train = data[data.index < pd.to_datetime("2018-01-01")]
    test = data[data.index >= pd.to_datetime("2018-01-01")]
    predictions = predict(train, test, predictors, model)
    return predictions

# Load or fetch S&P 500 data
sp500 = load_data()

# Prepare the data
sp500 = prepare_data(sp500)

# Feature Engineering
horizons = [2, 5, 60, 250, 1000]
for horizon in horizons:
    rolling_averages = sp500.rolling(window=horizon).mean()
    sp500[f"Close_Ratio_{horizon}"] = sp500["Close"] / rolling_averages["Close"]
    sp500[f"Trend_{horizon}"] = sp500.shift(1).rolling(window=horizon).sum()["Target"]

new_predictors = ["Close", "Volume", "Open", "High", "Low"] + [f"Close_Ratio_{h}" for h in horizons] + [f"Trend_{h}" for h in horizons]

# Ensure data is ready for model training
sp500.dropna(inplace=True)

# Train and predict
model = RandomForestClassifier(n_estimators=100, random_state=42)
predictions = backtest(sp500, model, new_predictors)

# Evaluation
print(predictions["Predictions"].value_counts())
print("Precision Score:", precision_score(predictions["Target"], predictions["Predictions"], zero_division=1))
