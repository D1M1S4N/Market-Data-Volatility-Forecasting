import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
# ¡Importante para las métricas reales!
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import matplotlib.pyplot as plt

# --- 1. Path Configuration (Local Version with pathlib) ---
# The script is located at '.../Market-Data-Volatility-Forecasting-GH/model/train.py'
# We need to go up two levels to reach the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / 'data' / 'preprocessed'
MODELS_DIR = PROJECT_ROOT / 'train_output'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
BASENAME = 'DUKASCOPY_EURUSD_15_2007-01-01_2025-01-01'

# ==============================================================================
# --- 2. QUASI-LIKELIHOOD (QL) LOSS FUNCTION ---
# ==============================================================================

def ql_loss(y_true, y_pred):
    """
    Calculates Quasi-Likelihood (QL) loss. Robust for volatility.
    L = (y_true / y_pred) - log(y_true / y_pred) - 1
    """
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    epsilon = K.epsilon()
    y_pred = tf.maximum(y_pred, epsilon)
    ratio = y_true / y_pred
    log_ratio = tf.math.log(tf.maximum(ratio, epsilon))
    loss = ratio - log_ratio - 1
    return K.mean(loss)

# ==============================================================================
# --- 3. CENTRALIZED HYPERPARAMETER CONFIGURATION (MODIFIED) ---
# ==============================================================================

class ModelConfig:
    """Model architecture parameters."""
    # Increased capacity to capture peaks
    LSTM_UNITS_1 = 128
    DROPOUT_1 = 0.3
    LSTM_UNITS_2 = 64
    DROPOUT_2 = 0.3
    DENSE_UNITS = 32
    OUTPUT_ACTIVATION = 'relu'

class TrainConfig:
    """Training process parameters."""
    LEARNING_RATE = 0.0005
    # KEY CHANGE! Using QL loss
    LOSS_FUNCTION = ql_loss
    METRICS = ['mae', 'mse']
    EPOCHS = 100
    BATCH_SIZE = 32
    EARLY_STOPPING_PATIENCE = 15 # A bit more patience
    SHUFFLE_TRAIN = False

# ==============================================================================

# --- 4. Load Preprocessed Data ---
print("\n--- Loading 3D preprocessed data... ---")
print("IMPORTANT! Assuming y_train, y_val, y_test are in REAL format (RV)")
try:
    X_train = np.load(DATA_DIR / f'{BASENAME}_X_train.npy')
    y_train = np.load(DATA_DIR / f'{BASENAME}_y_train.npy')
    X_val = np.load(DATA_DIR / f'{BASENAME}_X_val.npy')
    y_val = np.load(DATA_DIR / f'{BASENAME}_y_val.npy')
    X_test = np.load(DATA_DIR / f'{BASENAME}_X_test.npy')
    y_test = np.load(DATA_DIR / f'{BASENAME}_y_test.npy')
except FileNotFoundError:
    print(f"Error: .npy files not found in {DATA_DIR}")
    raise

N_TIMESTEPS = X_train.shape[1]
N_FEATURES = X_train.shape[2]
print(f"Data loaded: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"N_FEATURES should now be larger if PCA captured more families: {N_FEATURES}")


# ==============================================================================
# --- 5. Define Model Architecture (using Config) ---
# ==============================================================================

def build_lstm_model(input_shape):
    """
    Builds the Stacked Bidirectional Model using ModelConfig parameters.
    """
    inputs = layers.Input(shape=input_shape)

    # Layer 1: Bidirectional
    x = layers.Bidirectional(
        layers.LSTM(units=ModelConfig.LSTM_UNITS_1, return_sequences=True)
    )(inputs)
    x = layers.Dropout(ModelConfig.DROPOUT_1)(x)

    # Layer 2: Normal LSTM
    x = layers.LSTM(units=ModelConfig.LSTM_UNITS_2, return_sequences=False)(x)
    x = layers.Dropout(ModelConfig.DROPOUT_2)(x)

    # Dense Layer (Feed-forward)
    x = layers.Dense(units=ModelConfig.DENSE_UNITS, activation='relu')(x)

    # Output Layer
    outputs = layers.Dense(units=1, activation=ModelConfig.OUTPUT_ACTIVATION)(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model

# ==============================================================================


# Build the model
input_shape = (N_TIMESTEPS, N_FEATURES)
model = build_lstm_model(input_shape)

# --- 5. Compile the Model (using Config) ---
optimizer = tf.keras.optimizers.Adam(learning_rate=TrainConfig.LEARNING_RATE)

model.compile(optimizer=optimizer,
              loss=TrainConfig.LOSS_FUNCTION,
              metrics=TrainConfig.METRICS)
model.summary()

# --- 6. Train the Model (using Config) ---
print("\n--- Starting model training... ---")

model_checkpoint = ModelCheckpoint(
    filepath=MODELS_DIR / 'best_bilstm_vol_model_long_memory.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=TrainConfig.EARLY_STOPPING_PATIENCE,
    mode='min',
    verbose=1,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=TrainConfig.EPOCHS,
    batch_size=TrainConfig.BATCH_SIZE,
    callbacks=[model_checkpoint, early_stopping],
    shuffle=TrainConfig.SHUFFLE_TRAIN
)

print("--- Training finished. ---")

# ==============================================================================
# --- 7. Evaluate Model on Test Set ---
# ==============================================================================
print("\n--- Evaluating model on test set... ---")

test_loss, test_mae, test_mse = model.evaluate(X_test, y_test)
print(f"Test Loss (QL Loss): {test_loss:.6f}") # Main metric is now QL
print(f"Test Mean Absolute Error (MAE): {test_mae:.6f}")
print(f"Test Root Mean Squared Error (RMSE): {np.sqrt(test_mse):.6f}")


# --- 8. Plot Training History ---
print("\n--- Generating training history plot... ---")
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss (QL)')
plt.plot(history.history['val_loss'], label='Validation Loss (QL)')
plt.title('Training History: Model with QL Loss and Higher Capacity')
plt.xlabel('Epoch')
plt.ylabel('Quasi-Likelihood (QL) Loss')
plt.legend()
plt.grid(True)
plt.savefig(MODELS_DIR / 'training_history_ql_loss.png')
print(f"History plot saved at {MODELS_DIR / 'training_history_ql_loss.png'}")


print("\nTraining process completed. Best model saved at:")
print(f"{MODELS_DIR / 'best_bilstm_vol_model_long_memory.keras'}")