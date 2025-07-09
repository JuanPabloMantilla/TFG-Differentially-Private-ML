from utils import create_keras_model

import pandas as pd
import numpy as np
import pickle
import time
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import tensorflow as tf
from diffprivlib.mechanisms import GaussianAnalytic

def preprocess_data(raw_csv_path, processed_data_path, force_rerun=False):
    """
    Loads and preprocesses data from the raw CSV file.
    Saves the processed splits to a .npz file and returns them.
    If the .npz file already exists, it loads it directly to save time.
    """
    if os.path.exists(processed_data_path) and not force_rerun:
        print(f"Loading pre-processed data from {processed_data_path}...")
        with np.load(processed_data_path, allow_pickle=True) as data:
            return (data['X_train_std'], data['y_train'], data['X_test_std'], 
                    data['y_test'], data['X_train_norm'])

    print(f"Processing data from scratch from {raw_csv_path}...")
    
    # --- Data Loading and Initial Cleaning ---
    df = pd.read_csv(raw_csv_path)
    y_labels = df["los_class"].values
    
    features_df = df.drop(columns=[
        "los_class", 'hadm_id', 'subject_id', 'los', 'gender', 
        'first_careunit', 'admission_type', 'admission_location'
    ], errors='ignore')

    numeric_features_df = features_df.select_dtypes(include=np.number)
    numeric_features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # --- Dropping Columns (High Nulls & Zero Variance) ---
    percent_null = numeric_features_df.isnull().sum() / len(numeric_features_df)
    cols_to_drop_by_nulls = percent_null[percent_null >= 0.9].index
    numeric_features_df.drop(columns=cols_to_drop_by_nulls, inplace=True)

    # --- Imputation and Final Cleanup ---
    imputer = SimpleImputer(strategy='median')
    imputed_features_array = imputer.fit_transform(numeric_features_df)
    processed_features_df = pd.DataFrame(imputed_features_array, columns=numeric_features_df.columns)
    
    column_stds = processed_features_df.std()
    cols_to_drop_by_std = column_stds[column_stds == 0].index
    processed_features_df.drop(columns=cols_to_drop_by_std, inplace=True)

    if processed_features_df.empty:
        raise ValueError("No features remaining after preprocessing.")
    
    final_features_array = processed_features_df.values

    # --- Data Splitting and Scaling ---
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        final_features_array, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    
    # Standard scaling for model input
    scaler_standard = StandardScaler()
    X_train_std = scaler_standard.fit_transform(X_train_full)
    X_test_std = scaler_standard.transform(X_test)
    
    # MinMax scaling for noise calibration (maps data to [0,1] for sensitivity=1)
    scaler_min_max = MinMaxScaler()
    X_train_norm = scaler_min_max.fit_transform(X_train_std)

    # --- Saving Pre-processed Data ---
    np.savez(processed_data_path, 
             X_train_std=X_train_std, y_train=y_train_full,
             X_test_std=X_test_std, y_test=y_test,
             X_train_norm=X_train_norm,
             scaler_min_max_params={'scale': scaler_min_max.scale_, 'min': scaler_min_max.min_}
            )
    print(f"\nPreprocessed data saved to {processed_data_path}")
    
    return X_train_std, y_train_full, X_test_std, y_test, X_train_norm

def generate_noisy_datasets(X_train_norm, X_train_std, y_train, epsilon_values, delta):
    """
    Generates a list of datasets, including the baseline and noisy versions for each epsilon.
    """
    print("\nGenerating datasets for training...")
    datasets_to_train = [{
        "name": "Baseline (No DP)", "epsilon": np.inf,
        "X_train": X_train_std, "y_train": y_train
    }]
    
    # Recreate the MinMaxScaler used during preprocessing
    scaler_min_max = MinMaxScaler()
    scaler_min_max.fit(X_train_std) # Fit on the standardized data to learn the scale

    for epsilon in epsilon_values:
        print(f"Generating noisy dataset for epsilon = {epsilon:.2f}")
        gaussian_mechanism = GaussianAnalytic(epsilon=epsilon, delta=delta, sensitivity=1.0)
        
        # Apply noise to the normalized data
        X_noisy_normalized = gaussian_mechanism.randomise(X_train_norm)
        
        # Inverse transform noisy data back to the original standardized scale
        X_noisy_std = scaler_min_max.inverse_transform(X_noisy_normalized)
        
        datasets_to_train.append({
            "name": f"DP_Epsilon_{epsilon:.2f}", "epsilon": epsilon,
            "X_train": X_noisy_std, "y_train": y_train
        })
    print("All datasets generated.")
    return datasets_to_train

def train_and_evaluate_model(dataset_info, X_test, y_test, epochs, batch_size):
    """
    Trains and evaluates a Keras model on a single given dataset.
    """
    name = dataset_info["name"]
    epsilon = dataset_info["epsilon"]
    X_train = dataset_info["X_train"]
    y_train = dataset_info["y_train"]

    print(f"\n--- Processing dataset: {name} (Epsilon: {epsilon}) ---")

    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    # Truncate training data to be a multiple of batch size
    num_samples = (X_train.shape[0] // batch_size) * batch_size
    X_train_trunc = X_train[:num_samples]
    y_train_trunc = y_train[:num_samples]

    model = create_keras_model(n_features, n_classes)
    
    print(f"Training Keras model...")
    history = model.fit(
        X_train_trunc, y_train_trunc,
        epochs=epochs, batch_size=batch_size,
        validation_data=(X_test, y_test), 
        verbose=0
    )
    print("Training complete. Evaluating on test set...")

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred_proba = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_proba, axis=1)
    
    f1_macro = f1_score(y_test, y_pred_classes, average='macro', zero_division=0)
    try:
        auc_macro = roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='macro')
    except ValueError:
        auc_macro = np.nan
        
    print(f"Results: Test Acc={acc:.4f}, Test F1={f1_macro:.4f}, Test AUC={auc_macro:.4f}")
    
    return {
        "dataset_name": name, "epsilon": epsilon,
        "test_loss": float(loss), "test_accuracy": float(acc),
        "test_f1_macro": float(f1_macro), "test_auc_macro": float(auc_macro),
        "history": {k: [float(v) for v in val] for k, val in history.history.items()}
    }

def main():
    """
    Main function to orchestrate the entire experiment pipeline.
    """
    # --- Configuration ---
    RAW_CSV_PATH = '../data/variables_final.csv'
    PROCESSED_DATA_PATH = '../data/processed_data_dataset_perturb.npz' # Use a different cache file
    RESULTS_CSV_PATH = '../results/dataset_perturbation_results.csv'
    
    EPOCHS = 50
    BATCH_SIZE = 256
    DELTA = 1e-5
    EPSILON_VALUES = [0.1, 1.0, 2.5, 5.0, 7.27, 13.56, 20.0, 39.28, 100.0, 250.48]

    # --- Pipeline Execution ---
    # Step 1: Preprocess data
    X_train_std, y_train, X_test_std, y_test, X_train_norm = preprocess_data(RAW_CSV_PATH, PROCESSED_DATA_PATH)
    
    # Step 2: Generate all datasets (baseline + noisy versions)
    datasets_to_train = generate_noisy_datasets(X_train_norm, X_train_std, y_train, EPSILON_VALUES, DELTA)
    
    # Step 3: Train and evaluate a model for each dataset
    all_results = []
    script_start_time = time.time()
    
    for dataset_info in datasets_to_train:
        results = train_and_evaluate_model(dataset_info, X_test_std, y_test, EPOCHS, BATCH_SIZE)
        all_results.append(results)
        
    # --- Finalization and Saving ---
    script_end_time = time.time()
    total_time = script_end_time - script_start_time
    print(f"\nTotal execution time for all runs: {total_time:.2f} seconds ({total_time/60:.2f} minutes).")

    print(f"Saving all results to {RESULTS_CSV_PATH}...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_CSV_PATH, index=False)
    print("Process completed successfully.")

if __name__ == '__main__':
    main()