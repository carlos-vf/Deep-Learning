# src/tracker/visualize_performance.py
#
# A script to parse tracker performance files and generate visualizations
# to compare the results of different pipeline modes.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import re

def parse_performance_file(file_path):
    """
    Parses a motmetrics summary file to extract key performance indicators.
    This version is robust and handles different summary formats.
    
    Args:
        file_path (Path): The path to the performance .txt file.
        
    Returns:
        dict: A dictionary containing the parsed metrics, or None if parsing fails.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # --- FIX: Find the data line more robustly ---
        data_line = None
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        
        # The data line is typically the last line in the file.
        if lines:
            data_line = lines[-1]
        
        if not data_line:
            print(f"⚠️  Warning: Could not find any data lines in {file_path.name}.")
            return None

        # Use regular expressions to find all numbers (including percentages)
        values = re.findall(r'[\d\.]+\%?', data_line)
        
        # The first value is the name (e.g., 'gt_106' or 'overall'), which we don't need here.
        # We need at least 14 values (name + 13 metrics)
        if len(values) < 14:
            print(f"⚠️  Warning: Not enough metrics found in the summary line for {file_path.name}.")
            return None

        # Corrected indices for each metric based on the motmetrics table format
        metrics = {
            'IDF1': float(values[1].replace('%', '')),
            'IDP':  float(values[2].replace('%', '')),
            'IDR':  float(values[3].replace('%', '')),
            'Rcll': float(values[4].replace('%', '')),
            'Prcn': float(values[5].replace('%', '')),
            'FP':   int(values[9]),
            'FN':   int(values[10]),
            'IDs':  int(values[11]),
            'MOTA': float(values[13].replace('%', '')),
        }
        return metrics
    except Exception as e:
        print(f"⚠️  Warning: Could not parse file {file_path.name}. Error: {e}")
        return None

def plot_comparison(data, plot_title, output_dir):
    """
    Creates and saves a grouped bar chart comparing performance metrics.
    """
    df = pd.DataFrame(data)
    
    # Prepare data for plotting
    df_melted = df.melt(id_vars='Mode', var_name='Metric', value_name='Score')
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Create the grouped bar plot
    sns.barplot(data=df_melted, x='Metric', y='Score', hue='Mode', ax=ax, palette='viridis')
    
    # Add text labels on top of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=10, padding=3)
        
    ax.set_title(f'Tracker Performance Comparison for: {plot_title}', fontsize=18, fontweight='bold')
    ax.set_ylabel('Score / Count', fontsize=12)
    ax.set_xlabel('Performance Metric', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Pipeline Mode', fontsize=10)
    
    # Adjust y-axis limit for better visualization
    if not df_melted.empty:
        ax.set_ylim(0, max(df_melted['Score']) * 1.15)
    
    plt.tight_layout()
    
    # Save the plot with a clean filename
    safe_filename = plot_title.replace(" ", "_").lower()
    plot_path = output_dir / f"{safe_filename}_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"   -> Saved plot to {plot_path}")

def main():
    """
    Main function to find performance files, parse them, and generate plots.
    """
    parser = argparse.ArgumentParser(description="Visualize tracker performance metrics.")
    parser.add_argument("--perf-dir", required=True, help="Path to the top-level directory containing performance .txt files (e.g., 'src/tracker/performance/f4k').")
    parser.add_argument("--output-dir", default="outputs/plots", help="Directory to save the generated plots.")
    
    args = parser.parse_args()

    perf_dir = Path(args.perf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not perf_dir.is_dir():
        print(f"❌ Error: Performance directory not found at '{perf_dir}'")
        return

    all_files = perf_dir.rglob("*_performance.txt")
    perf_files = [f for f in all_files if not f.name.startswith('_')]
    
    if not perf_files:
        print(f"❌ Error: No individual performance files found recursively in '{perf_dir}'.")
        return

    # Group files by video name
    results_by_video = {}
    for f in perf_files:
        parts = f.stem.split('_')
        video_name = "_".join(parts[:-2])
        mode = parts[-2]
        
        if video_name not in results_by_video:
            results_by_video[video_name] = []
        
        metrics = parse_performance_file(f)
        if metrics:
            metrics['Mode'] = mode.capitalize()
            results_by_video[video_name].append(metrics)

    # Generate a plot for each video that has data for more than one mode
    print(f"\n📊 Generating individual performance plots...")
    for video_name, data in results_by_video.items():
        if len(data) > 1:
            plot_comparison(data, video_name, output_dir)
        else:
            print(f"   -> Skipping plot for {video_name} as it only has data for one mode.")
            
    # --- Generate an overall average performance plot ---
    print(f"\n📊 Generating overall average performance plot...")
    all_data_list = [item for sublist in results_by_video.values() for item in sublist]
    if all_data_list:
        overall_df = pd.DataFrame(all_data_list)
        average_perf = overall_df.groupby('Mode').mean().reset_index()
        if not average_perf.empty:
            plot_comparison(average_perf, "Overall Average", output_dir)
        else:
            print("   -> Skipping overall average plot due to no valid data.")
    else:
        print("   -> Skipping overall average plot due to no valid data.")

    print("\n✅ Visualization complete.")


if __name__ == '__main__':
    main()
