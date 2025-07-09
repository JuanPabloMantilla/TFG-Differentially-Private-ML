import pandas as pd
import numpy as np
import ast
import os

# --- Configuration ---
CSV_FILE_PATH = "../results/dataset_perturbation_results.csv" 
OUTPUT_TXT_FILE = "../results/dataset_perturbation_summary_table.txt"
TARGET_EPOCH = 20 # The epoch you want to get the validation accuracy from

# --- Load and Process Data ---
print(f"Loading data from {CSV_FILE_PATH}...")
try:
    results_df = pd.read_csv(CSV_FILE_PATH)
except FileNotFoundError:
    print(f"ERROR: File not found at '{CSV_FILE_PATH}'.")
    print("Please update the CSV_FILE_PATH variable in the script to the correct filename.")
    exit()

def process_dataframe(df):
    """Processes the dataframe to parse history and create consistent labels."""
    processed_rows = []
    
    # Handle the baseline model
    if 'Original_Standardized_Baseline' not in df['dataset_name'].unique() and np.isinf(df['input_data_noise_epsilon']).sum() == 0:
        baseline_placeholder = {'dataset_name': 'Original_Standardized_Baseline', 'input_data_noise_epsilon': np.inf, 
                                'test_f1_score': 0.6243, 'test_auc': 0.7211, 
                                'keras_training_history': None} # Add a placeholder history if you have one
        df = pd.concat([pd.DataFrame([baseline_placeholder]), df], ignore_index=True)

    for _, row in df.iterrows():
        history_data = row.get('keras_training_history', None)
        history_dict = None
        if isinstance(history_data, str):
            try:
                history_dict = ast.literal_eval(history_data)
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not parse history for row with epsilon {row['input_data_noise_epsilon']}: {e}.")
        elif isinstance(history_data, dict):
            history_dict = history_data
        
        new_row = row.to_dict()
        new_row['keras_training_history_dict'] = history_dict
        
        epsilon_val = row['input_data_noise_epsilon']
        new_row['epsilon_numeric'] = float(epsilon_val) # Store numeric value for sorting
        processed_rows.append(new_row)
    
    return pd.DataFrame(processed_rows)

results_df = process_dataframe(results_df)

# Sort by epsilon value to have a clean table order
results_df = results_df.sort_values(by='epsilon_numeric')

print("Data loaded and processed.")

# --- Extract Data for Table ---
table_data = []

for _, row in results_df.iterrows():
    epsilon_val = row['epsilon_numeric']
    
    # Label for the Epsilon column
    if np.isinf(epsilon_val):
        epsilon_label = "Non-DP (Baseline)"
    else:
        epsilon_label = f"{epsilon_val:g}" 
        
    # Get final F1 and AUC scores
    final_f1 = row['test_f1_score']
    final_auc = row['test_auc']
    
    # Get validation accuracy at TARGET_EPOCH
    accuracy_at_epoch = np.nan # Default to NaN
    history = row.get('keras_training_history_dict', None)
    
    if history and 'val_accuracy' in history:
        if len(history['val_accuracy']) >= TARGET_EPOCH:
            accuracy_at_epoch = history['val_accuracy'][TARGET_EPOCH - 1] # Index 19 for epoch 20
        else:
            pass # Keep it as NaN
            
    table_data.append({
        "Epsilon": epsilon_label,
        "Accuracy (Epoch 20)": accuracy_at_epoch,
        "Final F1-score": final_f1,
        "Final AUC": final_auc
    })

# --- Write Formatted Table to .txt File ---
print(f"Writing summary table to {OUTPUT_TXT_FILE}...")

try:
    with open(OUTPUT_TXT_FILE, 'w') as f:
        # Define column headers and calculate widths for alignment
        headers = list(table_data[0].keys())
        # Calculate max width for each column
        col_widths = {h: max(len(h), max(len(f"{d[h]:.4f}" if isinstance(d[h], float) else str(d[h])) for d in table_data)) for h in headers}
        
        # Write header
        header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
        f.write(header_line + "\n")
        
        # Write separator
        separator_line = "-|-".join('-' * col_widths[h] for h in headers)
        f.write(separator_line + "\n")
        
        # Write data rows
        for data_row in table_data:
            row_items = []
            for h in headers:
                value = data_row[h]
                if isinstance(value, float):
                    if np.isnan(value):
                        formatted_value = "N/A".ljust(col_widths[h])
                    else:
                        formatted_value = f"{value:.4f}".ljust(col_widths[h])
                else: # For Epsilon label string
                    formatted_value = str(value).ljust(col_widths[h])
                row_items.append(formatted_value)
            f.write(" | ".join(row_items) + "\n")
            
    print(f"Successfully created {OUTPUT_TXT_FILE}")

except Exception as e:
    print(f"An error occurred while writing the file: {e}")

print("\nProcess completed.")