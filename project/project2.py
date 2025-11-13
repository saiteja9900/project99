"""
app.py
- Simple Flask API to serve next-day prediction using saved models
- Endpoints:
    GET /predict?model=lr&ticker=AAPL
    returns JSON with predicted price (next day)
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model

app = Flask(__name__)

MODELS_DIR = "models"

# Load models on startup
LR_MODEL = None
LR_SCALER = None
LSTM_MODEL = None
LSTM_SCALER = None

def load_models():
    global LR_MODEL, LR_SCALER, LSTM_MODEL, LSTM_SCALER
    try:
        LR_MODEL = joblib.load(f"{MODELS_DIR}/linear_regression.joblib")
        LR_SCALER = joblib.load(f"{MODELS_DIR}/lr_scaler.joblib")
    except Exception as e:
        print("Linear model not found or problem loading:", e)
    try:
        LSTM_MODEL = load_model(f"{MODELS_DIR}/lstm_model.h5")
        LSTM_SCALER = joblib.load(f"{MODELS_DIR}/lstm_scaler.joblib")
    except Exception as e:
        print("LSTM model not found or problem loading:", e)

def fetch_latest(ticker, period="60d"):
    df = yf.download(ticker, period=period, progress=False)
    df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    df.rename(columns={'Adj Close':'Adj_Close'}, inplace=True)
    return df

@app.route("/predict", methods=["GET"])
def predict():
    ticker = request.args.get("ticker", "AAPL")
    model_name = request.args.get("model", "lr")  # 'lr' or 'lstm'
    # Fetch recent data
    df = fetch_latest(ticker, period="100d")
    if df.empty:
        return jsonify({"error":"no data fetched"}), 400

    # Very simple feature building for linear model: use lag_1..lag_3, MA5, MA20, Volatility20, Return
    df['Return'] = df['Adj_Close'].pct_change()
    df['LogReturn'] = np.log(df['Adj_Close']).diff()
    df['MA5'] = df['Adj_Close'].rolling(5).mean()
    df['MA20'] = df['Adj_Close'].rolling(20).mean()
    df['Volatility20'] = df['LogReturn'].rolling(20).std()
    for lag in range(1, 6):
        df[f'lag_{lag}'] = df['Adj_Close'].shift(lag)
    df = df.dropna()
    if df.empty:
        return jsonify({"error":"not enough recent data to compute features"}), 400

    last_row = df.iloc[-1]

    if model_name == 'lr':
        if LR_MODEL is None or LR_SCALER is None:
            return jsonify({"error":"linear model not loaded"}), 500
        features = ['lag_1', 'lag_2', 'lag_3', 'MA5', 'MA20', 'Volatility20', 'Return']
        X = last_row[features].values.reshape(1, -1)
        Xs = LR_SCALER.transform(X)
        pred = LR_MODEL.predict(Xs)[0]
        return jsonify({"ticker": ticker, "model": "linear_regression", "prediction": float(pred)})
    elif model_name == 'lstm':
        if LSTM_MODEL is None or LSTM_SCALER is None:
            return jsonify({"error":"lstm model not loaded"}), 500
        feature_cols = ['Adj_Close', 'MA5', 'MA20', 'Volatility20', 'Volume']
        # build combined scaled array for last look_back days
        look_back = 30
        sub = df[feature_cols].iloc[-look_back:].values
        if len(sub) < look_back:
            return jsonify({"error":"not enough history for LSTM prediction"}), 400
        scaled = LSTM_SCALER.transform(sub)
        X = scaled.reshape(1, scaled.shape[0], scaled.shape[1])
        pred_scaled = LSTM_MODEL.predict(X)[0][0]
        # inverse transform using scaler trick
        inv = np.zeros((1, len(feature_cols)))
        inv[0,0] = pred_scaled
        inv_pred = LSTM_SCALER.inverse_transform(inv)[0,0]
        return jsonify({"ticker": ticker, "model": "lstm", "prediction": float(inv_pred)})
    else:
        return jsonify({"error":"unknown model requested"}), 400

if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=5000, debug=True)
