import pandas as pd
import pickle

# --- Configuration ---
# The input file containing results from the training run (including F1 and AUC)
RESULTS_FILE = "../results/gradient_perturbation_results.pkl"
# The output text file for the final table
OUTPUT_TXT_FILE = "../results/gradient_perturbation_summary_table.txt"
# The specific epoch from which to pull validation accuracy
TARGET_EPOCH = 20

# --- Epsilon Mapping ---
# This dictionary maps the [noise_multiplier, l2_norm_clip] pair to the
# specific epsilon value you want to display in the table.
EPSILON_MAP = {
    (0.1, 0.5): 138990.3611,
    (0.1, 1.0): 138990.3611,
    (0.3, 0.5): 250.4863,
    (0.3, 1.0): 250.4863,
    (0.5, 0.5): 39.2856,
    (0.5, 1.0): 39.2856,
    (0.7, 0.5): 13.5634,
    (0.7, 1.0): 13.5634,
    (0.9, 0.5): 7.2799,
    (0.9, 1.0): 7.2799,
}

# --- Main Execution Block ---
if __name__ == '__main__':
    print(f"Loading results from '{RESULTS_FILE}'...")
    try:
        # Load the results file
        with open(RESULTS_FILE, 'rb') as f:
            results_list = pickle.load(f)
        print(f"  Successfully loaded {len(results_list)} model results.")
    except FileNotFoundError:
        print(f"  ERROR: The file '{RESULTS_FILE}' was not found.")
        exit()
    except Exception as e:
        print(f"  An error occurred while loading the file: {e}")
        exit()

    # --- Process the results to build the table data ---
    table_data = []
    for result in results_list:
        noise = result.get('noise_multiplier')
        clip = result.get('l2_norm_clip')
        
        # 1. Get Epsilon from the predefined map
        epsilon = EPSILON_MAP.get((noise, clip), 'N/A')

        # 2. Get Validation Accuracy at the target epoch
        history = result.get('history', {})
        val_accuracy_list = history.get('val_accuracy', [])
        accuracy_at_epoch_20 = 'N/A'
        # Check if the training ran for at least TARGET_EPOCH epochs
        if len(val_accuracy_list) >= TARGET_EPOCH:
            # List index is epoch - 1
            accuracy_at_epoch_20 = val_accuracy_list[TARGET_EPOCH - 1]
        else:
            print(f"  Warning: Model [N={noise}, L2={clip}] ran for fewer than {TARGET_EPOCH} epochs.")

        # 3. Get final F1-score
        f1_score = result.get('test_f1_macro', 'N/A')

        # 4. Get final AUC
        auc = result.get('test_auc_macro', 'N/A')

        # Append the processed data for this model to our list
        table_data.append({
            'Epsilon': epsilon,
            'Accuracy_Epoch_20': accuracy_at_epoch_20,
            'Final_F1_Score': f1_score,
            'Final_AUC': auc
        })

    # --- Create and Format the DataFrame ---
    if not table_data:
        print("No data processed. Exiting.")
        exit()
        
    # Convert list of dicts to a DataFrame
    final_df = pd.DataFrame(table_data)
    
    # Sort the table by Epsilon for a logical order
    final_df = final_df.sort_values(by='Epsilon', ascending=False).reset_index(drop=True)

    # Format numeric columns for better readability
    for col in ['Accuracy_Epoch_20', 'Final_F1_Score', 'Final_AUC']:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce').apply(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')
    
    # Format epsilon separately to handle large numbers
    final_df['Epsilon'] = pd.to_numeric(final_df['Epsilon'], errors='coerce').apply(lambda x: f"{x:,.4f}" if pd.notna(x) else 'N/A')
    
    # --- Save the table to a text file ---
    try:
        # Convert the formatted DataFrame to a string
        table_string = final_df.to_string(index=False)
        
        with open(OUTPUT_TXT_FILE, 'w') as f:
            f.write("--- Final Model Metrics Summary ---\n\n")
            f.write(table_string)
            
        print(f"\nSuccessfully saved the results table to '{OUTPUT_TXT_FILE}'")
        print("\n--- Table Preview ---")
        print(table_string)
        
    except Exception as e:
        print(f"\nAn error occurred while saving the file: {e}")