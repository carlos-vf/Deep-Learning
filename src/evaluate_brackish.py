# src/evaluate_tracker.py (Corrected Version)

import motmetrics as mm
import pandas as pd
from pathlib import Path
import argparse

def load_aau_ground_truth(file_path):
    """Loads and converts the AAU Bounding Box CSV format to a standard format."""
    df = pd.read_csv(file_path, delimiter=';')
    df['Width'] = df['Lower right corner X'] - df['Upper left corner X']
    df['Height'] = df['Lower right corner Y'] - df['Upper left corner Y']
    df['FrameId'] = df['Filename'].str.extract(r'-(\d+)\.png$').astype(int)
    df = df.rename(columns={'Object ID': 'Id', 'Upper left corner X': 'X', 'Upper left corner Y': 'Y'})
    df['Conf'] = 1.0
    return df

def evaluate_tracking(gt_file_path, ts_file_path, video_name):
    """Evaluates tracking performance for a single video."""
    gt_file_path = Path(gt_file_path)
    ts_file_path = Path(ts_file_path)

    if not gt_file_path.exists():
        print(f"❌ Error: Ground truth file not found at {gt_file_path}")
        return
    if not ts_file_path.exists():
        print(f"❌ Error: Tracker output file not found at {ts_file_path}")
        return

    # Load the full ground truth file
    full_gt = load_aau_ground_truth(gt_file_path)
    
    # *** KEY CHANGE: Filter the ground truth for the specific video ***
    gt = full_gt[full_gt['Filename'].str.contains(video_name)]
    
    if gt.empty:
        print(f"❌ Error: No ground truth data found for video name '{video_name}' in {gt_file_path}")
        return

    # Load tracker output (this is already for a single video)
    ts_col_names = ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Conf', 'classes', 'visibility', 'unused']
    ts = pd.read_csv(ts_file_path, header=None, names=ts_col_names)

    acc = mm.MOTAccumulator(auto_id=True)
    # Now loop through the frames of the *filtered* ground truth
    for frame_id in gt['FrameId'].unique():
        gt_frame = gt[gt['FrameId'] == frame_id]
        ts_frame = ts[ts['FrameId'] == frame_id]

        gt_ids = gt_frame['Id'].values
        ts_ids = ts_frame['Id'].values
        gt_boxes = gt_frame[['X', 'Y', 'Width', 'Height']].values
        ts_boxes = ts_frame[['X', 'Y', 'Width', 'Height']].values

        distance_matrix = mm.distances.iou_matrix(gt_boxes, ts_boxes, max_iou=0.5)
        
        acc.update(gt_ids, ts_ids, distance_matrix)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='overall')
    
    strsummary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    print(f"\n--- Tracking Metrics for video: {video_name} ---")
    print(strsummary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate tracking performance for a single video.")
    parser.add_argument("--gt", required=True, help="Path to the full ground truth file (e.g., test.csv).")
    parser.add_argument("--ts", required=True, help="Path to the tracker's specific output file (.txt).")
    parser.add_argument("--video-name", required=True, help="The base name of the video to filter by.")
    
    args = parser.parse_args()

    evaluate_tracking(args.gt, args.ts, args.video_name)