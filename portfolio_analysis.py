import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def download_data(tickers, start="2020-01-01", end=None):
    data = yf.download(tickers, start=start, end=end, group_by='ticker')
    return data


def build_features(df):
    features = []
    for ticker in df.columns.get_level_values(0).unique():
        stock = df[ticker].copy()
        stock['Return'] = stock['Close'].pct_change()
        stock['MA5'] = stock['Close'].rolling(window=5).mean()
        stock['MA10'] = stock['Close'].rolling(window=10).mean()
        stock['FutureClose'] = stock['Close'].shift(-1)
        stock['FutureReturn'] = stock['FutureClose'].pct_change()
        stock['Ticker'] = ticker
        features.append(stock)
    all_feat = pd.concat(features)
    all_feat.dropna(inplace=True)
    return all_feat


def train_model(features):
    X = features[['Close', 'MA5', 'MA10', 'Return']]
    y = features['FutureReturn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    score = r2_score(y_test, model.predict(X_test))
    return model, score


def predict_future(model, features):
    latest = features.groupby('Ticker').tail(1)
    preds = model.predict(latest[['Close', 'MA5', 'MA10', 'Return']])
    latest = latest.assign(Prediction=preds)
    top = latest.sort_values('Prediction', ascending=False)
    return top[['Ticker', 'Prediction']]


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    data = download_data(tickers)
    features = build_features(data)
    model, score = train_model(features)
    print(f"Model R^2 Score: {score:.4f}")
    top_stocks = predict_future(model, features)
    print("Top predicted stocks:")
    print(top_stocks.head())
