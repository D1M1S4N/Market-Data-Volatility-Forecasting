import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# --- 1. Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data' / 'preprocessed'
RAW_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'DUKASCOPY_EURUSD_15_2007-01-01_2025-01-01.csv'
MODELS_DIR = PROJECT_ROOT / 'train_output'
BASENAME = 'DUKASCOPY_EURUSD_15_2007-01-01_2025-01-01'
MODEL_PATH = MODELS_DIR / 'best_bilstm_vol_model_long_memory.keras'

# --- 2. Define ql_loss (Required to load the model) ---
def ql_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    epsilon = K.epsilon()
    y_pred = tf.maximum(y_pred, epsilon)
    ratio = y_true / y_pred
    log_ratio = tf.math.log(tf.maximum(ratio, epsilon))
    loss = ratio - log_ratio - 1
    return K.mean(loss)

# --- 3. Load Preprocessed Data (X, y) ---
print("--- Loading preprocessed data... ---")
X_val = np.load(DATA_DIR / f'{BASENAME}_X_val.npy')
y_val = np.load(DATA_DIR / f'{BASENAME}_y_val.npy')
X_test = np.load(DATA_DIR / f'{BASENAME}_X_test.npy')
y_test = np.load(DATA_DIR / f'{BASENAME}_y_test.npy')

print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# --- 4. Load Model ---
print(f"--- Loading model from {MODEL_PATH}... ---")
model = load_model(MODEL_PATH, custom_objects={'ql_loss': ql_loss})

# --- 5. Predictions ---
print("--- Making predictions... ---")
y_pred_val = model.predict(X_val).flatten()
y_pred_test = model.predict(X_test).flatten()

# --- 6. Plot 1: Prediction vs Reality (Validation) ---
print("--- Generating Validation plot... ---")
plt.figure(figsize=(14, 6))
plt.plot(y_val, label='Reality (RV)', color='blue', alpha=0.7)
plt.plot(y_pred_val, label='Prediction (RV)', color='orange', alpha=0.7, linestyle='--')
plt.title('Validation: Volatility Prediction vs Reality')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(MODELS_DIR / 'validation_prediction_vs_real.png')
print(f"Saved: {MODELS_DIR / 'validation_prediction_vs_real.png'}")

# --- 7. Prepare Data for Candlestick Plot (Test) ---
print("--- Processing raw data for candlestick plot... ---")

# Load raw data to get daily OHLC
df_raw = pd.read_csv(RAW_DATA_PATH)
# Assume first column is date
date_col = df_raw.columns[0]
df_raw[date_col] = pd.to_datetime(df_raw[date_col])
df_raw.set_index(date_col, inplace=True)

# Rename columns if necessary
df_raw.rename(columns=lambda x: x.strip().capitalize(), inplace=True)
# Basic mapping
col_map = {}
for c in df_raw.columns:
    if 'Open' in c: col_map[c] = 'Open'
    elif 'High' in c: col_map[c] = 'High'
    elif 'Low' in c: col_map[c] = 'Low'
    elif 'Close' in c: col_map[c] = 'Close'
df_raw.rename(columns=col_map, inplace=True)

# Resample to Daily OHLC
df_daily = df_raw.resample('D').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last'
}).dropna()

# Filter Test period
# In data_preprocessing.py: split_date_test = '2023-01-01'
TEST_START_DATE = '2023-01-01'
df_test_daily = df_daily[df_daily.index >= TEST_START_DATE].copy()

# Align predictions with days
# Prediction y_pred_test[i] corresponds to volatility of day i+1 of test set
# (due to shift(-1) of target in training)
# Therefore, we need days from index 1 onwards
df_plot = df_test_daily.iloc[1:].copy()

# Ensure equal lengths
min_len = min(len(df_plot), len(y_pred_test))
df_plot = df_plot.iloc[:min_len]
y_pred_plot = y_pred_test[:min_len]

print(f"Days to plot: {len(df_plot)}")

# Calculate Confidence Bands
# Predicted volatility is in % (e.g. 0.5 means 0.5%)
# Bands = Open * (1 +/- sigma/100)
sigma = y_pred_plot / 100

df_plot['Upper_1sigma'] = df_plot['Open'] * (1 + sigma)
df_plot['Lower_1sigma'] = df_plot['Open'] * (1 - sigma)
df_plot['Upper_2sigma'] = df_plot['Open'] * (1 + 2*sigma)
df_plot['Lower_2sigma'] = df_plot['Open'] * (1 - 2*sigma)

# --- 8. Plot 2: Candlestick with Confidence Bands (Zoom) ---
print("--- Generating Candlestick plot with Bands... ---")

# Zoom into first N days to see candles
N_ZOOM = 100
df_zoom = df_plot.iloc[:N_ZOOM]

fig, ax = plt.subplots(figsize=(18, 8))

# Draw candles manually
width = 0.6
width2 = 0.1

up = df_zoom[df_zoom.Close >= df_zoom.Open]
down = df_zoom[df_zoom.Close < df_zoom.Open]

# Bullish candles (Green)
ax.bar(up.index, up.Close - up.Open, width, bottom=up.Open, color='green', alpha=0.6)
ax.bar(up.index, up.High - up.Close, width2, bottom=up.Close, color='green', alpha=0.6)
ax.bar(up.index, up.Low - up.Open, width2, bottom=up.Open, color='green', alpha=0.6)

# Bearish candles (Red)
ax.bar(down.index, down.Close - down.Open, width, bottom=down.Open, color='red', alpha=0.6)
ax.bar(down.index, down.High - down.Open, width2, bottom=down.Open, color='red', alpha=0.6)
ax.bar(down.index, down.Low - down.Close, width2, bottom=down.Close, color='red', alpha=0.6)

# Draw Bands
ax.plot(df_zoom.index, df_zoom['Upper_1sigma'], color='blue', linestyle='--', linewidth=1, label='1-Sigma Band')
ax.plot(df_zoom.index, df_zoom['Lower_1sigma'], color='blue', linestyle='--', linewidth=1)

ax.plot(df_zoom.index, df_zoom['Upper_2sigma'], color='purple', linestyle=':', linewidth=1, label='2-Sigma Band')
ax.plot(df_zoom.index, df_zoom['Lower_2sigma'], color='purple', linestyle=':', linewidth=1)

ax.set_title(f'Visual Backtest: Volatility Prediction (First {N_ZOOM} days of Test)')
ax.set_ylabel('Price')
ax.legend()
ax.grid(True, alpha=0.2)

# Date format
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate()

plt.savefig(MODELS_DIR / 'test_candles_prediction.png')
print(f"Saved: {MODELS_DIR / 'test_candles_prediction.png'}")

print("\n--- Prediction process finished ---")
