# src/plotting_utils.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import ast
import os

def load_results(filepath):
    """
    Loads experiment results from a .pkl or .csv file.
    """
    print(f"Loading results from: {filepath}")
    _, extension = os.path.splitext(filepath)
    if extension == '.pkl':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif extension == '.csv':
        return pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file type: {extension}")

def process_history(history_data):
    """
    Safely parses a history string or dictionary into a dictionary.
    """
    if isinstance(history_data, str):
        try:
            return ast.literal_eval(history_data)
        except (ValueError, SyntaxError):
            return None
    elif isinstance(history_data, dict):
        return history_data
    return None

def plot_accuracy_comparison(title, output_path, baseline_history, dp_results, style_config):
    """
    Generates and saves a validation accuracy comparison plot.
    
    Args:
        title (str): The title of the plot.
        output_path (str): The path to save the plot image.
        baseline_history (dict): The history dictionary for the non-DP model.
        dp_results (list of dicts): A list of dictionaries, each for a DP model. 
                                    Each dict needs 'epsilon_label' and 'history' keys.
        style_config (dict): A dictionary with styling parameters.
    """
    print(f"Generating plot: {title}")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=style_config.get('figsize', (14, 8)))

    # Plot Non-DP baseline
    if baseline_history and 'val_accuracy' in baseline_history:
        val_acc = baseline_history['val_accuracy']
        ax.plot(range(1, len(val_acc) + 1), val_acc, 
                label='Non-DP (Baseline)', 
                color=style_config['baseline_color'], 
                linewidth=style_config['baseline_linewidth'], zorder=10)

    # Plot DP models
    colors = style_config['dp_colors']
    for i, result in enumerate(dp_results):
        history = result.get('history')
        if history and 'val_accuracy' in history:
            val_acc = history['val_accuracy']
            ax.plot(range(1, len(val_acc) + 1), val_acc, 
                    label=f"ε = {result['epsilon_label']:g}", 
                    color=colors[i % len(colors)], 
                    linewidth=style_config['dp_linewidth'])

    # Apply styling
    ax.set_title(title, fontsize=style_config['title_fontsize'])
    ax.set_xlabel('Epoch', fontsize=style_config['label_fontsize'])
    ax.set_ylabel('Validation Accuracy', fontsize=style_config['label_fontsize'])
    ax.tick_params(axis='both', which='major', labelsize=style_config['tick_fontsize'])
    ax.legend(loc='best', fontsize=style_config['legend_fontsize'])
    
    # Set axis limits
    if 'xlim' in style_config:
        ax.set_xlim(style_config['xlim'])
    if 'ylim' in style_config:
        ax.set_ylim(style_config['ylim'])

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to: {output_path}")

def generate_summary_table(title, output_path, table_data):
    """
    Generates and saves a formatted summary table to a .txt file.

    Args:
        title (str): The title for the summary.
        output_path (str): The path to save the .txt file.
        table_data (list of dicts): Data to be formatted into a table. 
                                     Keys should be column headers.
    """
    print(f"Generating summary table: {title}")
    if not table_data:
        print("Warning: No data provided for table generation.")
        return

    try:
        with open(output_path, 'w') as f:
            f.write(f"--- {title} ---\n\n")
            
            headers = list(table_data[0].keys())
            # Calculate max width for each column for alignment
            col_widths = {h: max(len(h), max(len(f"{d.get(h, ''):.4f}" if isinstance(d.get(h), float) else str(d.get(h, ''))) for d in table_data)) for h in headers}
            
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
                    value = data_row.get(h)
                    if isinstance(value, float):
                        formatted_value = f"{value:.4f}".ljust(col_widths[h])
                    else:
                        formatted_value = str(value).ljust(col_widths[h])
                    row_items.append(formatted_value)
                f.write(" | ".join(row_items) + "\n")
                
        print(f"Summary table saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred while writing the table file: {e}")