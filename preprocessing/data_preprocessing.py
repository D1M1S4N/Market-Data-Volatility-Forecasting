import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from typing import Dict, List
from pathlib import Path
import joblib

from utils import (
    load_csv,
    ensure_ohlcv_columns,
    iqr_filter,
    interpolate_gaps,
    atr,
    rsi,
    macd,
    bollinger_width,
    sma,
    ema,
    log_returns,
    rolling_stats,
    zscore_normalize,
    rolling_std,
    garman_klass_vol,
    downside_deviation,
    cci,
    adx,
    roc,
    log_volume_zscored,
    obv_slope,
    ad_line_slope,
    ma_slope,
    price_vs_ma,
    ema_cross_signal
)

RAW = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'DUKASCOPY_EURUSD_15_2007-01-01_2025-01-01.csv'
PRE = Path(__file__).resolve().parents[1] / 'data' / 'preprocessed'

ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

FAMILIES: Dict[str, List[str]] = {
    "volatility": ["rolling_std_", "atr_", "garman_klass_vol_", "downside_deviation_"],
    "momentum":    ["rsi_", "cci_", "adx_", "roc_"],
    "volume":     ["log_volume_zscored_", "obv_slope_", "ad_line_slope_"],
    "trend":   ["ma_slope_", "price_vs_ma_", "ema_cross_signal_"],
    "oscillator":   ["macd_", "ret_mean_", "ret_std_"] 
}

# PCA PIPELINE FUNCTIONS

def learn_pipeline_artefacts(df_train: pd.DataFrame, families: Dict[str, List[str]]) -> Dict:
    """Learn Imputers, Scalers, and PCAs from the train set and return them."""
    imputers = {}
    scalers = {}
    pcas = {}

    print("\nLearning PCA pipeline artifacts...")
    for fam_name, prefixes in tqdm(families.items(), desc="Learning PCA Families"):
        feature_list = [col for col in df_train.columns if any(col.startswith(p) for p in prefixes)]
        if not feature_list:
            print(f"  - Warning: No features found for family '{fam_name}'. Skipping.")
            continue

        X_train_family = df_train[feature_list]

        imputer = SimpleImputer(strategy="median").fit(X_train_family)
        imputers[fam_name] = imputer
        X_train_imputed = imputer.transform(X_train_family)

        scaler = RobustScaler().fit(X_train_imputed)
        scalers[fam_name] = scaler
        
        X_train_scaled = scaler.transform(X_train_imputed)
        pca = PCA(n_components=1, random_state=42).fit(X_train_scaled)
        pcas[fam_name] = pca

    print("  - PCA artifact learning complete.")
    return {"imputers": imputers, "scalers": scalers, "pcas": pcas, "families": families}

def apply_pipeline_transformations(df: pd.DataFrame, artefacts: Dict) -> pd.DataFrame:
    """Applies the transformers (artifacts) to a complete DataFrame."""
    imputers = artefacts["imputers"]
    scalers = artefacts["scalers"]
    pcas = artefacts["pcas"]
    families = artefacts["families"]
    
    df_out = pd.DataFrame(index=df.index)

    print("Applying learned PCA transformations...")
    for fam_name, _ in tqdm(families.items(), desc="Applying PCA Families"):
        if fam_name not in scalers: continue
            
        prefixes = families[fam_name]
        feature_list = [col for col in df.columns if any(col.startswith(p) for p in prefixes)]
        if not feature_list: continue

        X_family = df[feature_list]

        # Apply transformers in sequence: .transform() ONLY
        X_imputed = imputers[fam_name].transform(X_family)
        X_scaled = scalers[fam_name].transform(X_imputed)
        factor = pcas[fam_name].transform(X_scaled)
        
        # PCA columns are added here
        df_out[f'pca_{fam_name}'] = factor.flatten()
    
    return df_out

# LSTM SEQUENCE CREATION FUNCTION

def create_lstm_sequences(df: pd.DataFrame, feature_cols: list, target_col: str, n_steps: int):
    """
    Transform a 2D interval DataFrame into 3D daily sequences.
    """
    X, y = [], []
    
    for _, group in tqdm(df.resample('D'), desc="Creating 3D sequences"):
        if group.empty:
            continue
            
        features = group[feature_cols].values
        target = group[target_col].iloc[0]
        
        if pd.isna(target):
            continue
            
        if len(features) < n_steps:
            padding = np.zeros((n_steps - len(features), len(feature_cols)))
            features = np.concatenate([padding, features])
        elif len(features) > n_steps:
            features = features[-n_steps:]
            
        X.append(features)
        y.append(target)
        
    return np.array(X), np.array(y)

def preprocess_data_lstm_pca(raw_path: Path = RAW, out_dir: Path = PRE) -> Path:
    
    print("--- STEP 1: Loading and Cleaning ---")

    df = load_csv(str(raw_path))
    df = ensure_ohlcv_columns(df)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce') 
    df = iqr_filter(df, 'Close')
    df = interpolate_gaps(df)
    
    # Ensure 'Close' column exists
    if 'Close' not in df.columns:
        df['Close'] = df.get('Price', df[df.select_dtypes(include=[np.number]).columns[0]])

    print("--- STEP 2: Creating Target (Y) 'realized_volatility' ---")

    df['log_return'] = log_returns(df['Close'])
    df['squared_return'] = df['log_return']**2
    daily_rv = df['squared_return'].resample('D').sum().apply(np.sqrt)
    daily_rv = daily_rv * 100
    daily_rv.name = 'realized_volatility'
    df = df.merge(daily_rv, left_index=True, right_index=True, how='left')
    df['target'] = df['realized_volatility'].shift(-1)

    print("--- STEP 3: Creating Raw Features (X) ---")
    
    windows_short = [6, 55, 96]
    windows_medium = [480, 960]
    windows_long = [2016, 6048]

    windows = windows_short + windows_medium + windows_long
    print(f"Using {len(windows)} windows for features: {windows}")

    features = {}
    for w in tqdm(windows, desc="Processing feature windows"):
        features[f'atr_{w}'] = atr(df, window=w)
        features[f'rsi_{w}'] = rsi(df['Close'], window=w)
        ret_mean, ret_std = rolling_stats(df['log_return'].fillna(0), window=w)
        features[f'ret_mean_{w}'] = ret_mean
        features[f'ret_std_{w}'] = ret_std
        macd_line, signal_line, hist = macd(df['Close'], fast=w, slow=2*w, signal=w//2)
        features[f'macd_{w}'] = macd_line
        features[f'macd_signal_{w}'] = signal_line
        features[f'macd_hist_{w}'] = hist
        features[f'rolling_std_{w}'] = rolling_std(df['Close'], window=w)
        features[f'garman_klass_vol_{w}'] = garman_klass_vol(df, window=w)
        features[f'downside_deviation_{w}'] = downside_deviation(df['log_return'].fillna(0), window=w)
        features[f'cci_{w}'] = cci(df, window=w)
        features[f'adx_{w}'] = adx(df, window=w)
        features[f'roc_{w}'] = roc(df['Close'], window=w)
        features[f'log_volume_zscored_{w}'] = log_volume_zscored(df, window=w)
        features[f'obv_slope_{w}'] = obv_slope(df, window=w)
        features[f'ad_line_slope_{w}'] = ad_line_slope(df, window=w)
        features[f'ma_slope_{w}'] = ma_slope(df['Close'], window=w)
        features[f'price_vs_ma_{w}'] = price_vs_ma(df, window=w)
        features[f'ema_cross_signal_{w}'] = ema_cross_signal(df, window=w)

    df_features = pd.concat([df, pd.DataFrame(features)], axis=1)

    print("--- STEP 4: PCA Transformation Pipeline ---")

    out_dir.mkdir(parents=True, exist_ok=True)
    
    split_date_val = '2021-01-01'
    df_train_for_artifacts = df_features[df_features.index < split_date_val].copy()
    
    artefacts = learn_pipeline_artefacts(df_train_for_artifacts, FAMILIES)
    
    pca_artefacts_path = out_dir / (raw_path.stem + '_pca_pipeline.joblib')
    joblib.dump(artefacts, pca_artefacts_path)
    print(f"  - PCA artifacts saved at: {pca_artefacts_path}")
    
    df_pca = apply_pipeline_transformations(df_features, artefacts)
    
    df_final_2d = df_pca.merge(df_features[['target']], left_index=True, right_index=True, how='left')

    print("--- STEP 5: Final Scaling and 3D Sequencing ---")
    
    pca_feature_cols = [col for col in df_final_2d.columns if col.startswith('pca_')]
    target_col = 'target'
    
    df_cleaned = df_final_2d.dropna(subset=pca_feature_cols + [target_col])
    
    split_date_test = '2023-01-01'
    df_train = df_cleaned[df_cleaned.index < split_date_val].copy()
    df_val = df_cleaned[(df_cleaned.index >= split_date_val) & (df_cleaned.index < split_date_test)].copy()
    df_test = df_cleaned[df_cleaned.index >= split_date_test].copy()
    
    print(f"Train samples: {len(df_train)}, Val samples: {len(df_val)}, Test samples: {len(df_test)}")

    lstm_scaler = StandardScaler()
    lstm_scaler.fit(df_train[pca_feature_cols])

    df_train[pca_feature_cols] = lstm_scaler.transform(df_train[pca_feature_cols])
    df_val[pca_feature_cols] = lstm_scaler.transform(df_val[pca_feature_cols])
    df_test[pca_feature_cols] = lstm_scaler.transform(df_test[pca_feature_cols])

    lstm_scaler_path = out_dir / (raw_path.stem + '_lstm_scaler.joblib')
    joblib.dump(lstm_scaler, lstm_scaler_path)
    print(f"  - LSTM scaler saved at: {lstm_scaler_path}")

    N_STEPS = 96
    
    X_train, y_train = create_lstm_sequences(df_train, pca_feature_cols, target_col, N_STEPS)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    X_val, y_val = create_lstm_sequences(df_val, pca_feature_cols, target_col, N_STEPS)
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    X_test, y_test = create_lstm_sequences(df_test, pca_feature_cols, target_col, N_STEPS)
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    print("--- STEP 6: Saving final 3D arrays ---")
    
    np.save(out_dir / (raw_path.stem + '_X_train.npy'), X_train)
    np.save(out_dir / (raw_path.stem + '_y_train.npy'), y_train)
    np.save(out_dir / (raw_path.stem + '_X_val.npy'), X_val)
    np.save(out_dir / (raw_path.stem + '_y_val.npy'), y_val)
    np.save(out_dir / (raw_path.stem + '_X_test.npy'), X_test)
    np.save(out_dir / (raw_path.stem + '_y_test.npy'), y_test)
    
    print(f"\nPreprocessing with PCA complete. Artifacts saved in {out_dir}")
    
    return out_dir


if __name__ == "__main__":
    preprocess_data_lstm_pca()