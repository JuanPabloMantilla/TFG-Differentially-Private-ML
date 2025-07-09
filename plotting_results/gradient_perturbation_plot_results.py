import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import ast
import os

# --- Configuration ---
DP_RESULTS_PATH = "../results/gradient_perturbation_results.pkl"
NON_DP_RESULTS_PATH = "../results/dataset_perturbation_results.csv"
OUTPUT_DIR = "../results"
OUTPUT_FILENAME = "gradient_perturbation_accuracy_comparison.png"

# --- Plot Styling  ---
# Font Sizes
TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 20
LEGEND_FONTSIZE = 16
TICK_FONTSIZE = 16
# Line Widths
NON_DP_LINEWIDTH = 3.5
DP_LINEWIDTH = 3.0
# Colors
NON_DP_COLOR = 'black'
# Define a list of different "cold" colors for the DP models
DP_COLORS = ['#00008B', '#4169E1', '#20B2AA', '#87CEFA'] # DarkBlue, RoyalBlue, LightSeaGreen, LightSkyBlue
# Axis range and margin
X_AXIS_LIMIT = 20
Y_AXIS_BASE_MIN = 0.56
Y_AXIS_BASE_MAX = 0.70
Y_AXIS_MARGIN_PERCENT = 0.05 # 5% margin top and bottom

# --- DP Models to Plot ---
# This connects the [noise, clip] pair to the specific epsilon label for the legend.
MODELS_TO_PLOT = [
    {'noise': 0.9, 'clip': 0.5, 'epsilon_label': 7.27},
    {'noise': 0.7, 'clip': 0.5, 'epsilon_label': 13.54},
    {'noise': 0.5, 'clip': 0.5, 'epsilon_label': 39.28},
    {'noise': 0.3, 'clip': 0.5, 'epsilon_label': 250.48}
]

# --- Helper function ---
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# --- Main Execution Block ---
if __name__ == '__main__':
    ensure_dir(OUTPUT_DIR)

    # --- Load DP Model Data ---
    print(f"Loading DP model data from {DP_RESULTS_PATH}...")
    try:
        with open(DP_RESULTS_PATH, 'rb') as f:
            dp_results_list = pickle.load(f)
        dp_results_df = pd.DataFrame(dp_results_list)
        print(f"  Successfully loaded {len(dp_results_df)} DP model results.")
    except FileNotFoundError:
        print(f"  ERROR: File not found at '{DP_RESULTS_PATH}'. Exiting.")
        exit()

    # --- Load Non-DP Model Data ---
    print(f"Loading Non-DP model data from {NON_DP_RESULTS_PATH}...")
    non_dp_history = {}
    try:
        non_dp_df = pd.read_csv(NON_DP_RESULTS_PATH)
        baseline_row = non_dp_df[non_dp_df['dataset_name'] == 'Original_Standardized_Baseline']
        if not baseline_row.empty:
            history_str = baseline_row.iloc[0]['keras_training_history']
            non_dp_history = ast.literal_eval(history_str)
            print("  Successfully loaded and parsed Non-DP model history.")
        else:
            print("  Warning: Row for 'Original_Standardized_Baseline' not found. Baseline will not be plotted.")
    except Exception as e:
        print(f"  Warning: Could not load or parse Non-DP history. Baseline will not be plotted. Error: {e}")

    # --- Create the Plot ---
    print("\nGenerating final comparison plot...")
    fig, ax = plt.subplots(figsize=(14, 9))

    # 1. Plot Non-DP (Baseline) Model
    if non_dp_history and 'val_accuracy' in non_dp_history:
        val_accuracy = non_dp_history['val_accuracy']
        ax.plot(
            range(1, len(val_accuracy) + 1), val_accuracy,
            label='Non-DP (Baseline)',
            color=NON_DP_COLOR,
            linewidth=NON_DP_LINEWIDTH,
            linestyle='-' # Solid line as requested
        )

    # 2. Loop through the specified DP models and plot them
    for i, model_spec in enumerate(MODELS_TO_PLOT):
        noise = model_spec['noise']
        clip = model_spec['clip']
        epsilon_label = model_spec['epsilon_label']
        
        # Find the corresponding row in the DataFrame
        model_row = dp_results_df[
            (dp_results_df['noise_multiplier'] == noise) &
            (dp_results_df['l2_norm_clip'] == clip)
        ]

        if not model_row.empty:
            history = model_row.iloc[0].get('history', {})
            val_accuracy = history.get('val_accuracy', [])
            if val_accuracy:
                color = DP_COLORS[i % len(DP_COLORS)]
                
                ax.plot(
                    range(1, len(val_accuracy) + 1), val_accuracy,
                    label=f'ε = {epsilon_label}', # Use the provided epsilon for the label
                    color=color,
                    linewidth=DP_LINEWIDTH,
                    linestyle='-' # Solid line as requested
                )
            else:
                print(f"  Warning: History for DP model [N={noise}, L2={clip}] is empty or missing 'val_accuracy'.")
        else:
            print(f"  Warning: DP model [N={noise}, L2={clip}] not found in '{DP_RESULTS_PATH}'.")

    # 3. Apply Styling and Custom Axes
    ax.set_title('Validation Accuracy During Training', fontsize=TITLE_FONTSIZE)
    ax.set_xlabel('Epoch', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Validation Accuracy', fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    
    # Calculate Y-axis limits with margin
    base_range = Y_AXIS_BASE_MAX - Y_AXIS_BASE_MIN
    margin = base_range * Y_AXIS_MARGIN_PERCENT
    y_lim_bottom = Y_AXIS_BASE_MIN - margin
    y_lim_top = Y_AXIS_BASE_MAX + margin
    
    # Set custom axis limits
    ax.set_xlim(0, X_AXIS_LIMIT)
    ax.set_ylim(y_lim_bottom, y_lim_top)
    
    ax.legend(loc='best', fontsize=LEGEND_FONTSIZE)
    ax.grid(True, linestyle='--', alpha=0.6)

    # --- Save the Final Plot ---
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nFinal plot saved to {output_path}")
    print("\nScript finished.")