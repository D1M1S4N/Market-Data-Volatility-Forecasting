import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"
OUTPUT_DIR = PROJECT_ROOT / "train_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASENAME = "DUKASCOPY_EURUSD_15_2007-01-01_2025-01-01"


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


@dataclass
class TrainConfig:
    learning_rate: float = 0.0005
    epochs: int = 100
    batch_size: int = 32
    patience: int = 15
    shuffle_train: bool = False


@dataclass
class Experiment:
    name: str
    description: str
    build_fn: callable
    train_config: TrainConfig


def load_data():
    print("\n--- Loading 3D preprocessed data... ---")
    X_train = np.load(DATA_DIR / f"{BASENAME}_X_train.npy")
    y_train = np.load(DATA_DIR / f"{BASENAME}_y_train.npy")
    X_val = np.load(DATA_DIR / f"{BASENAME}_X_val.npy")
    y_val = np.load(DATA_DIR / f"{BASENAME}_y_val.npy")
    X_test = np.load(DATA_DIR / f"{BASENAME}_X_test.npy")
    y_test = np.load(DATA_DIR / f"{BASENAME}_y_test.npy")

    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]
    input_shape = (n_timesteps, n_features)

    print(
        "Data loaded: "
        f"X_train {X_train.shape}, y_train {y_train.shape}, "
        f"X_val {X_val.shape}, X_test {X_test.shape}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, input_shape


def build_baseline_model(input_shape, units=64, dropout=0.2):
    inputs = layers.Input(shape=input_shape)
    x = layers.Bidirectional(
        layers.LSTM(units=units, return_sequences=False)
    )(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="relu")(x)
    return keras.Model(inputs=inputs, outputs=outputs)


def build_stacked_model(input_shape, units_1=128, units_2=64, dropout=0.3):
    inputs = layers.Input(shape=input_shape)
    x = layers.Bidirectional(
        layers.LSTM(units=units_1, return_sequences=True)
    )(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(units=units_2, return_sequences=False)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="relu")(x)
    return keras.Model(inputs=inputs, outputs=outputs)


def build_attention_model(
    input_shape,
    units_1=128,
    units_2=64,
    dropout=0.3,
    l2_lambda=0.0,
):
    inputs = layers.Input(shape=input_shape)
    regularizer = keras.regularizers.l2(l2_lambda) if l2_lambda > 0 else None

    x = layers.Bidirectional(
        layers.LSTM(
            units=units_1,
            return_sequences=True,
            kernel_regularizer=regularizer,
        )
    )(inputs)
    x = layers.Dropout(dropout)(x)

    x = layers.LSTM(
        units=units_2,
        return_sequences=True,
        kernel_regularizer=regularizer,
    )(x)
    x = layers.Dropout(dropout)(x)

    key_dim = max(8, (x.shape[-1] or 64) // 4)
    attention_layer = layers.MultiHeadAttention(
        num_heads=2,
        key_dim=key_dim,
        name="self_attention",
    )
    attn_output, attn_scores = attention_layer(
        x, x, return_attention_scores=True
    )

    context = layers.GlobalAveragePooling1D()(attn_output)
    context = layers.Dense(32, activation="relu", kernel_regularizer=regularizer)(
        context
    )
    outputs = layers.Dense(1, activation="relu")(context)

    model = keras.Model(inputs=inputs, outputs=outputs)
    attention_extractor = keras.Model(inputs=inputs, outputs=attn_scores)

    return model, attention_extractor


def compile_model(model, train_config):
    optimizer = tf.keras.optimizers.Adam(learning_rate=train_config.learning_rate)
    model.compile(optimizer=optimizer, loss=ql_loss, metrics=["mae", "mse"])


def train_and_evaluate(
    experiment,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    input_shape,
):
    print(f"\n=== Running experiment: {experiment.name} ===")
    attention_extractor = None
    if experiment.name in {"attention", "regularized"}:
        model, attention_extractor = experiment.build_fn(input_shape)
    else:
        model = experiment.build_fn(input_shape)

    compile_model(model, experiment.train_config)
    model.summary()

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=OUTPUT_DIR / f"best_{experiment.name}_bilstm.keras",
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1,
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=experiment.train_config.patience,
        mode="min",
        verbose=1,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=experiment.train_config.epochs,
        batch_size=experiment.train_config.batch_size,
        callbacks=[model_checkpoint, early_stopping],
        shuffle=experiment.train_config.shuffle_train,
    )

    val_loss, val_mae, val_mse = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)

    history_path = OUTPUT_DIR / f"history_{experiment.name}.png"
    plt.figure(figsize=(12, 6))
    plt.plot(history.history["loss"], label="Training Loss (QL)")
    plt.plot(history.history["val_loss"], label="Validation Loss (QL)")
    plt.title(f"Training History: {experiment.description}")
    plt.xlabel("Epoch")
    plt.ylabel("Quasi-Likelihood (QL) Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(history_path)
    plt.close()
    print(f"Saved training history to {history_path}")

    attention_plot = None
    if attention_extractor is not None:
        attention_scores = attention_extractor.predict(X_val[:64], verbose=0)
        avg_attention = attention_scores.mean(axis=(0, 1, 2))

        attention_plot = OUTPUT_DIR / f"attention_weights_{experiment.name}.png"
        plt.figure(figsize=(12, 4))
        plt.plot(avg_attention, color="purple")
        plt.title(f"Average Attention Weights ({experiment.name})")
        plt.xlabel("Time Step")
        plt.ylabel("Average Weight")
        plt.grid(True, alpha=0.3)
        plt.savefig(attention_plot)
        plt.close()
        print(f"Saved attention weights to {attention_plot}")

    return {
        "experiment": experiment.name,
        "description": experiment.description,
        "val_loss": val_loss,
        "val_mae": val_mae,
        "val_rmse": float(np.sqrt(val_mse)),
        "test_loss": test_loss,
        "test_mae": test_mae,
        "test_rmse": float(np.sqrt(test_mse)),
        "epochs_trained": len(history.history["loss"]),
    }


def build_experiments():
    return [
        Experiment(
            name="baseline",
            description="Baseline single-layer BiLSTM",
            build_fn=build_baseline_model,
            train_config=TrainConfig(),
        ),
        Experiment(
            name="stacked",
            description="Stacked BiLSTM depth",
            build_fn=build_stacked_model,
            train_config=TrainConfig(),
        ),
        Experiment(
            name="attention",
            description="Stacked BiLSTM with self-attention",
            build_fn=lambda input_shape: build_attention_model(
                input_shape,
                dropout=0.3,
                l2_lambda=0.0,
            ),
            train_config=TrainConfig(),
        ),
        Experiment(
            name="regularized",
            description="Attention model with dropout + L2",
            build_fn=lambda input_shape: build_attention_model(
                input_shape,
                dropout=0.4,
                l2_lambda=1e-4,
            ),
            train_config=TrainConfig(patience=20),
        ),
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ablation experiments for BiLSTM volatility forecasting."
    )
    parser.add_argument(
        "--only",
        choices=["baseline", "stacked", "attention", "regularized"],
        help="Run a single experiment by name.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    experiments = build_experiments()
    if args.only:
        experiments = [exp for exp in experiments if exp.name == args.only]

    X_train, y_train, X_val, y_val, X_test, y_test, input_shape = load_data()

    results = []
    for experiment in experiments:
        results.append(
            train_and_evaluate(
                experiment,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                input_shape,
            )
        )

    results_path = OUTPUT_DIR / "ablation_results.csv"
    header = (
        "experiment,description,val_loss,val_mae,val_rmse,"
        "test_loss,test_mae,test_rmse,epochs_trained"
    )
    rows = [
        ",".join(
            [
                result["experiment"],
                f'"{result["description"]}"',
                f"{result["val_loss"]:.6f}",
                f"{result["val_mae"]:.6f}",
                f"{result["val_rmse"]:.6f}",
                f"{result["test_loss"]:.6f}",
                f"{result["test_mae"]:.6f}",
                f"{result["test_rmse"]:.6f}",
                str(result["epochs_trained"]),
            ]
        )
        for result in results
    ]

    results_path.write_text("\n".join([header, *rows]))
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
