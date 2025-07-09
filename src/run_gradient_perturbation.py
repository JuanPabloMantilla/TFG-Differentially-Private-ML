from utils import create_keras_model

import pandas as pd
import numpy as np
import pickle
import time
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

import tensorflow as tf
import tensorflow_privacy as tfp
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy

def preprocess_data(raw_csv_path, processed_data_path, force_rerun=False):
    """
    Loads data from the original CSV, preprocesses it, and saves it as a .npz file.
    If the .npz file already exists, it loads it directly to save time.
    """
    if os.path.exists(processed_data_path) and not force_rerun:
        print(f"Loading pre-processed data from {processed_data_path}...")
        with np.load(processed_data_path, allow_pickle=True) as data:
            return data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test']

    print(f"Processing data from scratch from {raw_csv_path}...")
    
    # --- Load and Initial Cleanup ---
    df = pd.read_csv(raw_csv_path)
    target_column_name = "los_class"
    y = df[target_column_name].values
    
    columns_to_drop_for_X_initial = [
        target_column_name, 'hadm_id', 'subject_id', 'los', 'gender', 
        'first_careunit', 'admission_type', 'admission_location'
    ]
    X_df = df.drop(columns=columns_to_drop_for_X_initial)

    # --- Step 1: Identify and remove columns with more than 90% NaN ---
    print("\nStep 1: Checking for columns with >90% NaN...")
    nan_threshold = 0.90
    null_ratio = X_df.isnull().sum() / len(X_df)
    cols_to_drop = null_ratio[null_ratio > nan_threshold].index.tolist()
    if cols_to_drop:
        print(f"Found {len(cols_to_drop)} columns with more than {nan_threshold:.0%} NaN values to remove.")
        X_df = X_df.drop(columns=cols_to_drop)
    else:
        print(f"No columns found with more than {nan_threshold:.0%} NaN values.")

    # --- Step 2: Handle remaining NaNs and Infs in X_df ---
    print("\nStep 2: Imputing remaining NaNs and Infs...")
    for col in X_df.columns:
        if pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = X_df[col].replace([np.inf, -np.inf], 0)
            if X_df[col].isnull().any():
                mean_val = X_df[col].mean()
                X_df[col] = X_df[col].fillna(mean_val)

    # --- Step 3: Check for columns with zero or near-zero standard deviation ---
    print("\nStep 3: Checking for columns with near-zero standard deviation...")
    numeric_cols_for_std_check = X_df.select_dtypes(include=np.number).columns
    if not numeric_cols_for_std_check.empty:
        column_stds = X_df[numeric_cols_for_std_check].std()
        std_threshold = 1e-9 
        columns_with_near_zero_std = column_stds[column_stds < std_threshold].index.tolist()
        if columns_with_near_zero_std:
            print(f"Found {len(columns_with_near_zero_std)} constant columns to remove.")
            X_df = X_df.drop(columns=columns_with_near_zero_std)
        else:
            print("No constant columns found.")

    X = X_df.values
    print(f"Final number of features: {X.shape[1]}")

    # --- Scaling and Splitting Data ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    # --- Saving Pre-processed Data ---
    np.savez(processed_data_path, 
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test)
    print(f"\nPreprocessed data saved to {processed_data_path}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_evaluate_dp_model(X_train, y_train, X_val, y_val, X_test, y_test, noise_multiplier, l2_norm_clip, epochs, batch_size, delta):
    """
    Defines, trains, and evaluates a DP model for a specific hyperparameter configuration.
    """
    print(f"\n--- Training with noise_multiplier={noise_multiplier}, l2_norm_clip={l2_norm_clip} ---")
    
    # --- Truncate data to be a multiple of batch_size ---
    num_train_samples = X_train.shape[0]
    samples_to_keep = (num_train_samples // batch_size) * batch_size
    if num_train_samples > samples_to_keep:
        X_train_trunc = X_train[:samples_to_keep]
        y_train_trunc = y_train[:samples_to_keep]
    else:
        X_train_trunc, y_train_trunc = X_train, y_train
    
    n_features = X_train_trunc.shape[1]
    n_classes = len(np.unique(y_train_trunc))
    
    # --- Model Definition and Compilation ---
    model = create_keras_model(n_features, n_classes)

    optimizer = tfp.DPKerasAdamOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=batch_size,
        learning_rate=0.001
    )

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # --- Model Training ---
    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        X_train_trunc, y_train_trunc,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        verbose=2
    )

    # --- Epsilon Calculation ---
    epsilon_tuple = compute_dp_sgd_privacy(
        n=X_train_trunc.shape[0],
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        epochs=epochs,
        delta=delta,
    )
    epsilon_value = epsilon_tuple[0]
    print(f"Calculated Epsilon (ε): {epsilon_value:.4f}")

    # --- Model Evaluation ---
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_proba, axis=1)

    try:
        auc_macro = roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='macro')
    except ValueError as e:
        print(f"Could not calculate AUC: {e}. Setting AUC to 0.")
        auc_macro = 0.0
    
    f1_macro = f1_score(y_test, y_pred_classes, average='macro', zero_division=0)
    print(f"Test Accuracy: {test_accuracy:.4f}, Test AUC (Macro OVO): {auc_macro:.4f}, Test F1-score (Macro): {f1_macro:.4f}")

    # --- Collect Results ---
    run_results = {
        'noise_multiplier': noise_multiplier, 'l2_norm_clip': l2_norm_clip,
        'epsilon': epsilon_value, 'history': history.history,
        'test_loss': test_loss, 'test_accuracy': test_accuracy,
        'test_auc_macro': auc_macro, 'test_f1_macro': f1_macro
    }
    return run_results

def main():
    """
    Main function to orchestrate the preprocessing and the training loop.
    """
    # --- Paths and Hyperparameters ---
    RAW_CSV_PATH = '../data/variables_final.csv'
    PROCESSED_DATA_PATH = '../data/processed_data.npz'
    RESULTS_PKL_PATH = '../results/gradient_perturbation_results.pkl'
    
    EPOCHS = 50
    BATCH_SIZE = 256
    DELTA = 1e-5
    
    # DP hyperparameters to iterate over
    noise_multipliers = [0.1, 0.3, 0.5, 0.7, 0.9] 
    l2_norm_clips = [0.5, 1.0]

    # --- Pipeline Execution ---
    # Step 1: Preprocess data (or load if it already exists)
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(RAW_CSV_PATH, PROCESSED_DATA_PATH)

    # Step 2: Training and evaluation loop
    all_results = []
    script_start_time = time.time()
    
    for noise in noise_multipliers:
        for clip in l2_norm_clips:
            result = train_evaluate_dp_model(
                X_train, y_train, X_val, y_val, X_test, y_test,
                noise_multiplier=noise, l2_norm_clip=clip,
                epochs=EPOCHS, batch_size=BATCH_SIZE, delta=DELTA
            )
            all_results.append(result)

    # --- Finalization and Saving ---
    script_end_time = time.time()
    total_time = script_end_time - script_start_time
    print(f"\nTotal execution time for all runs: {total_time:.2f} seconds ({total_time/60:.2f} minutes).")

    print(f"Saving all results to {RESULTS_PKL_PATH}...")
    try:
        with open(RESULTS_PKL_PATH, 'wb') as f:
            pickle.dump(all_results, f)
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == '__main__':
    main()