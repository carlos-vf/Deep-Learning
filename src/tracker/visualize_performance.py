# src/tracker/visualize_performance.py
#
# A script to parse tracker performance files and generate visualizations
# including bar charts, radar charts, and heatmaps.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import re
import numpy as np

def parse_performance_file(file_path):
    """
    Parses a motmetrics summary file to extract key performance indicators.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        data_line = None
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if lines:
            # The data line is the last line of the file
            data_line = lines[-1]
        
        if not data_line:
            return None

        # This regex finds all numbers, percentages, and words in the line
        values = re.findall(r'([a-zA-Z_]+|[\d\.]+\%?)', data_line)
        
        # This is robust to changes in column order or spacing.
        header_line = None
        for line in lines:
            if line.strip().startswith('IDF1'):
                header_line = line.strip().split()
                break
        
        if not header_line:
            return None

        # The first value in the data line is the name (e.g., 'overall')
        data_values = data_line.split()[1:]
        
        # Create a dictionary mapping the header to its value
        metrics_dict = dict(zip(header_line, data_values))

        metrics = {
            'IDF1': float(metrics_dict['IDF1'].replace('%', '')),
            'IDP':  float(metrics_dict['IDP'].replace('%', '')),
            'IDR':  float(metrics_dict['IDR'].replace('%', '')),
            'Rcll': float(metrics_dict['Rcll'].replace('%', '')),
            'Prcn': float(metrics_dict['Prcn'].replace('%', '')),
            'FP':   int(metrics_dict['FP']),
            'FN':   int(metrics_dict['FN']),
            'IDs':  int(metrics_dict['IDs']),
            'MOTA': float(metrics_dict['MOTA'].replace('%', '')),
        }
        return metrics
    except Exception as e:
        print(f"⚠️  Warning: Could not parse file {file_path.name}. Error: {e}")
        return None

def plot_bar_comparison(data, plot_title, output_dir):
    """
    Creates and saves a grouped bar chart comparing percentage-based performance metrics.
    """
    df = pd.DataFrame(data)
    
    # --- Filter for percentage-based metrics only ---
    percentage_metrics = ['Mode', 'MOTA', 'IDF1', 'Rcll', 'Prcn', 'IDP', 'IDR']
    df_filtered = df[percentage_metrics]
    
    # Prepare data for plotting
    df_melted = df_filtered.melt(id_vars='Mode', var_name='Metric', value_name='Score')
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))
    
    sns.barplot(data=df_melted, x='Metric', y='Score', hue='Mode', ax=ax, palette='viridis')
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', fontsize=10, padding=3) # Added '%' to label
        
    ax.set_title(f'Tracker Performance Comparison for: {plot_title}', fontsize=18, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12) # Updated Y-axis label
    ax.set_xlabel('Performance Metric', fontsize=12)
    ax.tick_params(axis='x', rotation=0) # Set rotation to 0 for better readability
    ax.legend(title='Pipeline Mode', fontsize=10)
    
    # Set y-axis to a 0-100 scale for percentages
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    
    safe_filename = plot_title.replace(" ", "_").lower()
    plot_path = output_dir / f"{safe_filename}_bar_chart.png"
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"   -> Saved Bar Chart to {plot_path}")


def plot_radar_chart(data, plot_title, output_dir):
    """
    Creates and saves a high-quality radar chart comparing key percentage-based metrics.
    """
    df = pd.DataFrame(data)
    
    metrics = ['MOTA', 'IDF1', 'Rcll', 'Prcn', 'IDP', 'IDR']
    df_filtered = df[metrics + ['Mode']]
    df_melted = df_filtered.melt(id_vars='Mode', var_name='Metric', value_name='Score')

    pivot_df = df_melted.pivot(index='Mode', columns='Metric', values='Score')
    pivot_df = pivot_df[metrics]

    labels = pivot_df.columns
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    
    cmap = plt.colormaps.get_cmap("Set2")
    colors = cmap(np.linspace(0, 1, len(pivot_df)))

    for i, (index, row) in enumerate(pivot_df.iterrows()):
        values = row.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=colors[i], linewidth=2, linestyle='solid', label=index, marker='o')
        ax.fill(angles, values, color=colors[i], alpha=0.25)

    ax.set_ylim(0, 100)
    
    # --- THE FIX IS HERE ---
    # 1. Set the grid line positions
    ax.set_rgrids([25, 50, 75])
    # 2. Set the labels for the grid lines and their style
    ax.set_yticklabels(["25%", "50%", "75%"], color="grey", size=10)
    # 3. Style the grid lines themselves separately
    ax.yaxis.grid(True, linestyle='--', color='grey', alpha=0.8)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12)
    
    plt.title(plot_title, size=20, color='black', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    safe_filename = plot_title.replace(" ", "_").lower()
    plot_path = output_dir / f"{safe_filename}_radar_chart.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   -> Saved Radar Chart to {plot_path}")

def plot_heatmap(data, mode, output_dir):
    """
    Creates and saves a heatmap showing performance across all videos for a specific mode.
    """
    df = pd.DataFrame(data)
    df_mode = df[df['Mode'] == mode]
    
    if df_mode.empty:
        print(f"   -> No data for '{mode}' mode, skipping heatmap.")
        return
        
    pivot_df = df_mode.set_index('Video').drop(columns='Mode')
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="viridis", linewidths=.5, ax=ax)
    
    ax.set_title(f'Performance Heatmap for {mode} Mode', fontsize=16, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Video', fontsize=12)
    
    plt.tight_layout()
    
    plot_path = output_dir / f"{mode.lower()}_performance_heatmap.png"
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"   -> Saved Heatmap to {plot_path}")

def main():
    """
    Main function to find performance files, parse them, and generate plots.
    """
    parser = argparse.ArgumentParser(description="Visualize tracker performance metrics.")
    parser.add_argument("--perf-dir", required=True, help="Path to the directory containing performance .txt files.")
    parser.add_argument("--output-dir", default="outputs/plots", help="Directory to save the generated plots.")
    
    args = parser.parse_args()

    perf_dir = Path(args.perf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not perf_dir.is_dir():
        print(f"❌ Error: Performance directory not found at '{perf_dir}'")
        return

    # Process individual video reports
    all_files = perf_dir.rglob("*_performance.txt")
    individual_files = [f for f in all_files if '_overall_' not in f.name]
    
    results_by_video = []
    if individual_files:
        for f in individual_files:
            parts = f.stem.split('_')
            video_name = "_".join(parts[:-2])
            mode = parts[-2]
            metrics = parse_performance_file(f)
            if metrics:
                metrics['Mode'] = mode.capitalize()
                metrics['Video'] = video_name
                results_by_video.append(metrics)

    # Process overall summary reports
    summary_files = list(perf_dir.glob("*_overall_performance_summary.txt"))
    overall_data = []
    if summary_files:
        for f in summary_files:
            base_name = f.stem.split('_overall')[0]
            mode = base_name.split('_')[-1]
            metrics = parse_performance_file(f)
            if metrics:
                metrics['Mode'] = mode.capitalize()
                overall_data.append(metrics)
    
    # --- Generate Plots ---
    if results_by_video:
        print(f"\n📊 Generating individual performance heatmaps...")
        df_individual = pd.DataFrame(results_by_video)
        for mode in df_individual['Mode'].unique():
            plot_heatmap(df_individual, mode, output_dir)

    if overall_data:
        print(f"\n📊 Generating overall performance plots...")
        plot_bar_comparison(overall_data, "Overall Performance", output_dir)
        plot_radar_chart(overall_data, "Overall Performance Profile", output_dir)

    print("\n✅ Visualization complete.")


if __name__ == '__main__':
    main()
