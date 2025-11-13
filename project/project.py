"""
train_and_eval.py
- Downloads historical stock data with yfinance
- Preprocesses and creates features
- Trains Linear Regression (sklearn)
- Trains LSTM (tensorflow.keras)
- Plots predicted vs actual
- Saves trained models and scalers
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -----------------------
# Config
# -----------------------
TICKER = "AAPL"            # change as needed
START = "2015-01-01"
END = None                 # None means today
DATA_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------
# Utility functions
# -----------------------
def download_stock(ticker, start=START, end=END):
    print(f"Downloading {ticker} from {start} to {end or 'today'}...")
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        raise RuntimeError("No data downloaded. Check ticker or network.")
    data = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    data.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)
    return data

def add_features(df):
    # Create lag features, returns, moving averages, volatility
    df = df.copy()
    df['Return'] = df['Adj_Close'].pct_change()
    df['LogReturn'] = np.log(df['Adj_Close']).diff()
    df['MA5'] = df['Adj_Close'].rolling(5).mean()
    df['MA20'] = df['Adj_Close'].rolling(20).mean()
    df['Volatility20'] = df['LogReturn'].rolling(20).std()
    # Lags of close
    for lag in range(1, 6):
        df[f'lag_{lag}'] = df['Adj_Close'].shift(lag)
    # Drop NaNs
    df = df.dropna()
    return df

def train_test_split_time_series(df, test_ratio=0.2):
    n = len(df)
    test_size = int(n * test_ratio)
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]
    return train, test

# -----------------------
# Linear Regression pipeline
# -----------------------
def train_linear_regression(train_df, test_df, features, target='Adj_Close'):
    X_train = train_df[features].values
    y_train = train_df[target].values
    X_test = test_df[features].values
    y_test = test_df[target].values

    # scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    preds = lr.predict(X_test_scaled)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print("LinearRegression MSE:", mse, "R2:", r2)

    # Save model and scaler
    joblib.dump(lr, os.path.join(DATA_DIR, 'linear_regression.joblib'))
    joblib.dump(scaler, os.path.join(DATA_DIR, 'lr_scaler.joblib'))

    return preds, y_test

# -----------------------
# LSTM pipeline
# -----------------------
def create_sequences(values, look_back=30):
    """
    values: numpy array shape (n_samples, n_features)
    returns sequences and corresponding targets (next-step forecasting)
    """
    X, y = [], []
    for i in range(len(values) - look_back):
        X.append(values[i:i+look_back])
        y.append(values[i+look_back, 0])  # assume first column is target (Adj_Close scaled)
    return np.array(X), np.array(y)

def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model

def train_lstm(train_df, test_df, feature_cols, target_col='Adj_Close', look_back=30, epochs=50, batch_size=32):
    # Use MinMaxScaler for LSTM inputs
    scaler = MinMaxScaler()
    combined = pd.concat([train_df[feature_cols], test_df[feature_cols]], axis=0)
    scaler.fit(combined.values)

    train_scaled = scaler.transform(train_df[feature_cols].values)
    test_scaled = scaler.transform(test_df[feature_cols].values)

    # build sequences
    X_train, y_train = create_sequences(train_scaled, look_back=look_back)
    # For test, we need sequences that may overlap from end of train (we'll use combined to make proper sequences)
    combined_scaled = scaler.transform(combined.values)
    X_combined, y_combined = create_sequences(combined_scaled, look_back=look_back)

    # Determine index where test begins in combined sequences
    n_train = len(train_df)
    # The sequences in combined correspond to start indices 0..len(combined)-look_back-1
    # The first sequence that lies fully in test set starts at index n_train - look_back
    test_start_seq_idx = max(0, n_train - look_back)
    X_test = X_combined[test_start_seq_idx:]
    y_test = y_combined[test_start_seq_idx:]

    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    model = build_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))
    checkpoint_path = os.path.join(DATA_DIR, 'best_lstm.h5')
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    mc = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es, mc],
        verbose=2
    )

    # Predictions
    preds_scaled = model.predict(X_test)
    # preds_scaled is scaled in target space because target was scaled as first feature; to inverse transform we need to reconstruct
    # Create a placeholder array to inverse transform using scaler: put preds in first column and other features zeros
    inv_input = np.zeros((len(preds_scaled), len(feature_cols)))
    inv_input[:, 0] = preds_scaled[:, 0]
    inv_pred = scaler.inverse_transform(inv_input)[:, 0]

    inv_y = np.zeros((len(y_test), len(feature_cols)))
    inv_y[:, 0] = y_test
    inv_y_vals = scaler.inverse_transform(inv_y)[:, 0]

    # Save model and scaler
    model.save(os.path.join(DATA_DIR, 'lstm_model.h5'))
    joblib.dump(scaler, os.path.join(DATA_DIR, 'lstm_scaler.joblib'))

    mse = mean_squared_error(inv_y_vals, inv_pred)
    r2 = r2_score(inv_y_vals, inv_pred)
    print("LSTM MSE:", mse, "R2:", r2)

    return inv_pred, inv_y_vals, history

# -----------------------
# Plotting
# -----------------------
def plot_preds(dates_test, actual, pred_lr=None, pred_lstm=None, title_add=""):
    plt.figure(figsize=(12,6))
    plt.plot(dates_test, actual, label='Actual', linewidth=2)
    if pred_lr is not None:
        plt.plot(dates_test, pred_lr, label='LinearRegression Pred', linestyle='--')
    if pred_lstm is not None:
        plt.plot(dates_test[:len(pred_lstm)], pred_lstm, label='LSTM Pred', linestyle=':')
    plt.xlabel('Date')
    plt.ylabel('Adj Close')
    plt.title(f'Actual vs Predicted {title_add}')
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, f'pred_vs_actual_{TICKER}.png')
    plt.savefig(fname)
    print("Saved plot to:", fname)
    plt.close()

# -----------------------
# Main
# -----------------------
def main():
    df = download_stock(TICKER, START, END)
    df_feat = add_features(df)
    train_df, test_df = train_test_split_time_series(df_feat, test_ratio=0.2)

    # Features for Linear Regression (choose lag features and MAs)
    features_lr = ['lag_1', 'lag_2', 'lag_3', 'MA5', 'MA20', 'Volatility20', 'Return']
    # Ensure features exist
    for f in features_lr:
        if f not in train_df.columns:
            raise RuntimeError(f"Feature {f} not present in data")

    lr_preds, lr_actual = train_linear_regression(train_df, test_df, features_lr, target='Adj_Close')

    # LSTM features (we'll include Adj_Close as first column, and some others)
    feature_cols_lstm = ['Adj_Close', 'MA5', 'MA20', 'Volatility20', 'Volume']
    # Drop rows with NaNs again just to be safe
    train_df_lstm = train_df[feature_cols_lstm].dropna()
    test_df_lstm = test_df[feature_cols_lstm].dropna()

    lstm_pred, lstm_actual, history = train_lstm(train_df_lstm, test_df_lstm, feature_cols_lstm,
                                                 target_col='Adj_Close', look_back=30, epochs=50, batch_size=32)

    # Create plotting dates - for LR predictions the entire test_df index; for LSTM predictions it's shorter
    dates_test = test_df.index
    plot_preds(dates_test, test_df['Adj_Close'].values, pred_lr=lr_preds, pred_lstm=lstm_pred, title_add=TICKER)

    # Print last rows: actual vs preds (align lengths)
    print("\n--- LR sample ---")
    for i in range(5):
        idx = -1 - i
        print("Date:", dates_test[idx].date(), "Actual:", lr_actual[idx], "LR_Pred:", lr_preds[idx])

    print("\n--- LSTM sample ---")
    # LSTM test predictions correspond to sequences starting from test start. Print last 5
    for i in range(5):
        print("Actual:", lstm_actual[-1-i], "LSTM_Pred:", lstm_pred[-1-i])

if __name__ == "__main__":
    main()
