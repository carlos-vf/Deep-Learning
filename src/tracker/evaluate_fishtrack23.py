# src/evaluate_tracker.py (for JSON ground truth)

import motmetrics as mm
import pandas as pd
from pathlib import Path
import argparse
import json

def load_json_ground_truth(file_path):
    """
    Parses the specific JSON annotation format and converts it into a
    pandas DataFrame suitable for motmetrics.
    """
    print(f"Parsing ground truth file: {file_path.name}")
    all_gt_data = []
    
    with open(file_path, 'r') as f:
        data = json.load(f)

    # The main data is under the "tracks" key
    tracks = data.get('tracks', {})
    
    # Iterate through each tracked object in the JSON
    for track_id, track_info in tracks.items():
        # Get the list of frames where this object appears
        features = track_info.get('features', [])
        
        for feature in features:
            frame_id = feature.get('frame')
            bounds = feature.get('bounds')
            
            if frame_id is not None and bounds and len(bounds) == 4:
                x1, y1, x2, y2 = bounds
                width = x2 - x1
                height = y2 - y1
                
                # Append the data in the format motmetrics expects
                all_gt_data.append([
                    frame_id,
                    int(track_id),
                    x1,
                    y1,
                    width,
                    height,
                    1.0 # Ground truth confidence is always 1.0
                ])

    # Create the final DataFrame
    gt_df = pd.DataFrame(all_gt_data, columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Conf'])
    return gt_df


def evaluate_tracking(gt_file_path, ts_file_path):
    """
    Evaluates tracking performance using the py-motmetrics library.
    """
    gt_file_path = Path(gt_file_path)
    ts_file_path = Path(ts_file_path)

    if not gt_file_path.exists() or not ts_file_path.exists():
        print("❌ Error: Ground truth or tracker file not found.")
        return

    # Load ground truth and tracker output using our new JSON parser
    gt = load_json_ground_truth(gt_file_path)
    
    ts_col_names = ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Conf', 'classes', 'visibility', 'unused']
    ts = pd.read_csv(ts_file_path, header=None, names=ts_col_names)

    if gt.empty:
        print("❌ Error: No valid ground truth data could be loaded from the JSON file.")
        return

    acc = mm.MOTAccumulator(auto_id=True)

    # Loop through each frame that has a ground truth annotation
    for frame_id in gt['FrameId'].unique():
        gt_frame = gt[gt['FrameId'] == frame_id]
        ts_frame = ts[ts['FrameId'] == frame_id]

        gt_ids = gt_frame['Id'].values
        ts_ids = ts_frame['Id'].values
        gt_boxes = gt_frame[['X', 'Y', 'Width', 'Height']].values
        ts_boxes = ts_frame[['X', 'Y', 'Width', 'Height']].values

        distance_matrix = mm.distances.iou_matrix(gt_boxes, ts_boxes, max_iou=0.5)
        
        acc.update(gt_ids, ts_ids, distance_matrix)

    # Compute and display the metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='overall')
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print("\n--- Tracking Metrics ---")
    print(strsummary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate tracking performance against JSON ground truth.")
    parser.add_argument("--gt-json", required=True, help="Path to the ground truth JSON file.")
    parser.add_argument("--ts-txt", required=True, help="Path to the tracker's output text file.")
    
    args = parser.parse_args()

    evaluate_tracking(args.gt_json, args.ts_txt)