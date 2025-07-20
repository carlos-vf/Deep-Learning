# src/tracker/evaluate_f4k.py
# A script to evaluate tracker performance across an entire folder of F4K results.
# It saves individual reports for each video and a final overall summary.

import xml.etree.ElementTree as ET
import motmetrics as mm
import pandas as pd
from pathlib import Path
import argparse
import numpy as np

def load_f4k_ground_truth(xml_file):
    """
    Parses a single F4K XML file and returns a DataFrame suitable for motmetrics.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    key_frames_element = root.find('keyFrames')
    key_frames = set()

    if key_frames_element is not None:
        for kf_range in key_frames_element.findall('keyFrame'):
            start = int(kf_range.attrib['start'])
            end = int(kf_range.attrib['end'])
            for i in range(start, end + 1):
                key_frames.add(i)

    data = []
    for frame in root.findall('frame'):
        frame_id = int(frame.attrib['id'])
        if frame_id not in key_frames:
            continue

        for obj in frame.findall('object'):
            obj_type = obj.attrib.get('objectType')
            if 'fish' not in obj_type:
                continue
                
            track_id = int(obj.attrib['trackingId'])
            
            contour_element = obj.find('contour')
            if contour_element is None or contour_element.text is None:
                continue

            point_pairs = contour_element.text.strip().split(',')
            points = [list(map(int, pair.split())) for pair in point_pairs if pair]
            
            if not points:
                continue

            points = np.array(points)
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            width = x_max - x_min
            height = y_max - y_min

            data.append([frame_id, track_id, x_min, y_min, width, height])

    return pd.DataFrame(data, columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height'])

def main():
    """
    Main function to run batch evaluation on a folder of tracker results.
    """
    parser = argparse.ArgumentParser(description="Evaluate tracking performance on the F4K dataset for multiple videos.")
    parser.add_argument("--gt-dir", required=True, help="Path to the folder containing ground truth XML files.")
    parser.add_argument("--ts-dir", required=True, help="Path to the folder containing the tracker's output .txt files.")
    parser.add_argument("--output-dir", default="src/tracker/performance/f4k", 
                        help="Directory to save the performance results.")
    
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    ts_dir = Path(args.ts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # This accumulator will store the combined results from all videos
    acc_main = mm.MOTAccumulator(auto_id=True)
    
    gt_files = sorted(list(gt_dir.glob("*.xml")))
    if not gt_files:
        print(f"❌ Error: No ground truth XML files found in '{gt_dir}'")
        return

    print(f"Found {len(gt_files)} ground truth files. Starting evaluation...")
    mh = mm.metrics.create()

    # Loop through each ground truth file
    for gt_xml_path in gt_files:
        video_name = gt_xml_path.stem
        ts_txt_path = next(ts_dir.glob(f"{video_name}*.txt"), None)

        if ts_txt_path is None:
            print(f"⚠️ Warning: No tracker results file found for '{video_name}'. Skipping.")
            continue
        
        print(f"\n--- Evaluating: {video_name} ---")

        gt = load_f4k_ground_truth(gt_xml_path)
        ts_col_names = ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Conf', 'classes', 'visibility', 'unused']
        ts = pd.read_csv(ts_txt_path, header=None, names=ts_col_names)

        # --- NEW: Create a temporary accumulator for this video only ---
        acc_video = mm.MOTAccumulator(auto_id=True)

        for frame_id in gt['FrameId'].unique():
            gt_frame = gt[gt['FrameId'] == frame_id]
            ts_frame = ts[ts['FrameId'] == frame_id]

            gt_ids = gt_frame['Id'].values
            ts_ids = ts_frame['Id'].values
            gt_boxes = gt_frame[['X', 'Y', 'Width', 'Height']].values
            ts_boxes = ts_frame[['X', 'Y', 'Width', 'Height']].values

            distance_matrix = mm.distances.iou_matrix(gt_boxes, ts_boxes, max_iou=0.5)
            
            # Update BOTH accumulators: one for the overall total, one for this video only.
            acc_main.update(gt_ids, ts_ids, distance_matrix)
            acc_video.update(gt_ids, ts_ids, distance_matrix)

        # --- NEW: Compute and save the summary for the individual video ---
        summary_video = mh.compute(acc_video, metrics=mm.metrics.motchallenge_metrics, name=video_name)
        strsummary_video = mm.io.render_summary(
            summary_video,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        
        pipeline_mode = "unknown"
        filename_parts = ts_txt_path.stem.split('_')
        if len(filename_parts) > 1:
            # Assumes format like 'gt_113_buffered_results'
            pipeline_mode = filename_parts[-2]

        header_text_video = f"--- Tracking Metrics for video: {video_name} (Mode: {pipeline_mode.capitalize()}) ---"
        print(header_text_video)
        print(strsummary_video)

        output_file_path_video = output_dir / f"{video_name}_{pipeline_mode}_performance.txt"
        with open(output_file_path_video, 'w') as f:
            f.write(f"{header_text_video}\n\n")
            f.write(strsummary_video)
        print(f"📄 Individual performance saved to: {output_file_path_video}")

    # --- After processing all videos, compute and display the final overall summary ---
    summary_overall = mh.compute(acc_main, metrics=mm.metrics.motchallenge_metrics, name='overall')
    
    strsummary_overall = mm.io.render_summary(
        summary_overall,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    
    header_text_overall = "--- Overall Tracking Performance Summary (All Videos) ---"
    print(f"\n{header_text_overall}")
    print(strsummary_overall)

    # Save the final summary report
    mode = str(ts_dir).split("\\")[-1]
    print(f"Mode: {mode}")
    summary_file_path = output_dir / ("f4k_" + f"{mode}" + "_overall_performance_summary.txt")
    with open(summary_file_path, 'w') as f:
        f.write(f"{header_text_overall}\n\n")
        f.write(strsummary_overall)
    
    print(f"\n📄 Overall performance summary saved to: {summary_file_path}")
    print("✅ Batch evaluation complete.")


if __name__ == '__main__':
    main()