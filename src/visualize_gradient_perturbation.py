# visualize_gradient_perturbation.py

from src.plotting_utils import load_results, process_history, plot_accuracy_comparison, generate_summary_table
import numpy as np
import pandas as pd

# --- Configuration ---
DP_RESULTS_PATH = "results/gradient_perturbation_results.pkl"
NON_DP_RESULTS_PATH = "results/dataset_perturbation_results.csv" # For baseline
TARGET_EPOCH = 20
MODELS_TO_PLOT = [
    {'epsilon_label': 7.27, 'noise_multiplier': 0.9, 'l2_norm_clip': 0.5},
    {'epsilon_label': 13.56, 'noise_multiplier': 0.7, 'l2_norm_clip': 0.5},
    {'epsilon_label': 39.28, 'noise_multiplier': 0.5, 'l2_norm_clip': 0.5},
    {'epsilon_label': 250.48, 'noise_multiplier': 0.3, 'l2_norm_clip': 0.5}
]

# Plotting Style
style_config = {
    'figsize': (14, 8), 'title_fontsize': 21, 'label_fontsize': 18,
    'legend_fontsize': 16, 'tick_fontsize': 14,
    'baseline_linewidth': 4.0, 'dp_linewidth': 3.4,
    'baseline_color': 'black',
    'dp_colors': ['#00008B', '#4169E1', '#20B2AA', '#87CEFA'], # Cold colors
    'xlim': (0, 20), 'ylim': (0.55, 0.71)
}

# --- Main ---
if __name__ == "__main__":
    # Load DP and Non-DP data
    dp_results = load_results(DP_RESULTS_PATH)
    non_dp_df = load_results(NON_DP_RESULTS_PATH)
    
    # Extract baseline history
    baseline_row = non_dp_df[np.isinf(non_dp_df['input_data_noise_epsilon'])].iloc[0]
    baseline_history = process_history(baseline_row['keras_training_history'])

    # Prepare DP data for plotting
    dp_plot_data = []
    for spec in MODELS_TO_PLOT:
        for result in dp_results:
            if result['noise_multiplier'] == spec['noise_multiplier'] and result['l2_norm_clip'] == spec['l2_norm_clip']:
                dp_plot_data.append({'epsilon_label': spec['epsilon_label'], 'history': result['history']})
                break
    
    # Generate Plot
    plot_accuracy_comparison(
        title='Validation Accuracy During Training (Gradient Perturbation)',
        output_path='results/gradient_perturbation_accuracy.png',
        baseline_history=baseline_history,
        dp_results=dp_plot_data,
        style_config=style_config
    )

    # Prepare data for table
    table_data = []
    for result in dp_results:
        history = result['history']
        acc_epoch_20 = history['val_accuracy'][TARGET_EPOCH - 1] if history and len(history.get('val_accuracy', [])) >= TARGET_EPOCH else np.nan
        table_data.append({
            'Epsilon': result['epsilon'],
            'Accuracy (Epoch 20)': acc_epoch_20,
            'Final F1-Score': result['test_f1_macro'],
            'Final AUC': result['test_auc_macro']
        })

    # Generate Table
    generate_summary_table(
        title="Gradient Perturbation Model Metrics",
        output_path="results/gradient_perturbation_summary.txt",
        table_data=sorted(table_data, key=lambda x: x['Epsilon'])
    )