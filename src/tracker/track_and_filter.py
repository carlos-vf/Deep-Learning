# src/track_and_filter.py (with tracker config parameter)

import cv2
from ultralytics import YOLO
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def run_tracker_filter(model_path, video_path, output_dir, tracker_config_path, min_track_duration=5):
    """
    Performs a two-pass process using a single-class model. It first collects
    all track data using a specified config file and then filters for stable tracks.
    """
    # --- 1. SETUP ---
    model_path = Path(model_path)
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    tracker_config_path = Path(tracker_config_path)
    output_dir.mkdir(exist_ok=True)
    video_output_path = output_dir / f"{video_path.stem}_filtered.mp4"
    txt_output_path = output_dir / f"{video_path.stem}_filtered_results.txt"

    if not all([model_path.exists(), video_path.exists(), tracker_config_path.exists()]):
        print("❌ Error: Model, video, or tracker config file not found.")
        return

    # --- 2. LOAD MODEL AND SETUP ---
    print(f"✅ Loading model from: {model_path}")
    model = YOLO(model_path)
    
    np.random.seed(42)
    colors = np.random.randint(0, 256, size=(100, 3), dtype=np.uint8)

    cap = cv2.VideoCapture(str(video_path))
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(str(video_output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # --- 3. FIRST PASS: DATA COLLECTION ---
    print(f"⏳ Pass 1/2: Collecting all raw tracking data with config '{tracker_config_path.name}'...")
    all_tracks_data = []
    
    # Use the reliable high-level model.track() function
    results_generator = model.track(
        source=str(video_path), 
        stream=True, 
        persist=True,
        conf=0.05,
        tracker=str(tracker_config_path) # Use the config file path from arguments
    )

    for frame_number, results in enumerate(results_generator, 1):
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist()
            confs = results.boxes.conf.cpu().numpy()
            
            for box, track_id, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = box
                all_tracks_data.append([frame_number, track_id, x1, y1, x2, y2, conf])

    print(f"✅ Pass 1 complete. Found {len(all_tracks_data)} raw detections.")
    cap.release()

    # --- 4. FILTERING PASS & 5. SECOND PASS ---
    if not all_tracks_data:
        print("No tracks found, exiting.")
        video_writer.release()
        return
        
    print(f"⏳ Pass 2/2: Filtering for stable tracks (min duration: {min_track_duration} frames)...")
    tracks_df = pd.DataFrame(all_tracks_data, columns=['frame_id', 'track_id', 'x1', 'y1', 'x2', 'y2', 'conf'])
    track_durations = tracks_df.groupby('track_id').size()
    stable_track_ids = track_durations[track_durations >= min_track_duration].index
    stable_tracks_df = tracks_df[tracks_df['track_id'].isin(stable_track_ids)]
    
    with open(txt_output_path, "w") as f:
        for _, row in stable_tracks_df.iterrows():
            width = row['x2'] - row['x1']
            height = row['y2'] - row['y1']
            f.write(f"{int(row['frame_id'])},{int(row['track_id'])},{row['x1']:.2f},{row['y1']:.2f},{width:.2f},{height:.2f},{row['conf']:.2f},-1,-1,-1\n")

    cap = cv2.VideoCapture(str(video_path))
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        frame_tracks = stable_tracks_df[stable_tracks_df['frame_id'] == frame_number]
        
        for _, track in frame_tracks.iterrows():
            x1, y1, x2, y2 = map(int, [track['x1'], track['y1'], track['x2'], track['y2']])
            track_id = int(track['track_id'])
            
            label_text = f"id:{track_id} fish {track['conf']:.2f}"
            color = tuple(colors[track_id % 100].tolist())
            
            (w_text, h_text), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w_text, y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
        video_writer.write(frame)

    # --- 6. CLEANUP ---
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print("-" * 30)
    print(f"✅ Filtering complete.")
    print(f"📄 Filtered tracking data saved to: {txt_output_path}")
    print(f"🎥 Filtered video saved to: {video_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tracker-based confidence boosting.")
    parser.add_argument("--model", default="models/best.pt", help="Path to the single-class YOLO model.")
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--output-dir", default="phase3_outputs", help="Directory to save the results.")
    parser.add_argument("--tracker-config", default="./src/tracker/bytetrack.yaml", help="Path to the tracker's YAML configuration file.")
    parser.add_argument("--min-duration", type=int, default=5, help="Minimum number of frames a track must exist to be considered stable.")
    
    args = parser.parse_args()
    run_tracker_filter(args.model, args.video, args.output_dir, args.tracker_config, args.min_duration)