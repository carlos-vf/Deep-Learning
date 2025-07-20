import xml.etree.ElementTree as ET
import motmetrics as mm
import pandas as pd
from pathlib import Path
import argparse
import numpy as np

import xml.etree.ElementTree as ET
import motmetrics as mm
import pandas as pd
from pathlib import Path
import argparse
import numpy as np

def load_f4k_ground_truth(xml_file):
    """
    Parses the F4K XML file, correctly handling keyFrame ranges and contour points,
    and returns a DataFrame suitable for motmetrics.
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

    if not key_frames:
        print("Warning: No key frames found in XML file.")

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

    df = pd.DataFrame(data, columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height'])
    return df

def evaluate_f4k_tracking(gt_xml_path, ts_txt_path, output_dir):
    """
    Compares a tracker's output text file against the F4K ground truth
    and saves the results to a text file.
    """
    gt_xml_path = Path(gt_xml_path)
    ts_txt_path = Path(ts_txt_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not gt_xml_path.exists() or not ts_txt_path.exists():
        print("❌ Error: Ground truth or tracker file not found.")
        return

    pipeline_mode = "unknown"
    filename_parts = ts_txt_path.stem.split('_')
    if len(filename_parts) > 1 and "results" in filename_parts[-1]:
        pipeline_mode = filename_parts[-2]

    gt = load_f4k_ground_truth(gt_xml_path)
    if gt.empty:
        print("❌ Error: No valid ground truth data could be loaded for key frames.")
        return

    ts_col_names = ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Conf', 'classes', 'visibility', 'unused']
    full_ts = pd.read_csv(ts_txt_path, header=None, names=ts_col_names)

    key_frame_ids = gt['FrameId'].unique()
    ts = full_ts[full_ts['FrameId'].isin(key_frame_ids)]

    acc = mm.MOTAccumulator(auto_id=True)
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
    
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    
    # Print results to console with the mode included
    header_text = f"--- Tracking Metrics for video: {gt_xml_path.stem} (Mode: {pipeline_mode.capitalize()}) ---"
    print(f"\n{header_text}")
    print(f"(Evaluated on {len(key_frame_ids)} key frames only)")
    print(strsummary)

    # Save results to file with the mode included
    output_file_path = output_dir / f"{gt_xml_path.stem}_{pipeline_mode}_performance.txt"
    with open(output_file_path, 'w') as f:
        f.write(f"{header_text}\n")
        f.write(f"(Evaluated on {len(key_frame_ids)} key frames only)\n\n")
        f.write(strsummary)
    
    print(f"\n📄 Performance metrics saved to: {output_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate tracking on the F4K dataset.")
    parser.add_argument("--gt-xml", required=True, help="Path to the ground truth XML file.")
    parser.add_argument("--ts-txt", required=True, help="Path to the tracker's output .txt file.")
    parser.add_argument("--output-dir", default="src/tracker/performance", 
                        help="Directory to save the performance results.")
    
    args = parser.parse_args()
    evaluate_f4k_tracking(args.gt_xml, args.ts_txt, args.output_dir)
