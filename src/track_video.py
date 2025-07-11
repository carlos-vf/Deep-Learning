# src/track_video.py (Updated Version)

import cv2
from ultralytics import YOLO
from pathlib import Path
import time
import argparse # Import the argparse library

def run_tracker(model_path, video_path, output_dir):
    model_path = Path(model_path)
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Define a unique path for the tracker output text file based on the video name
    txt_output_path = output_dir / f"{video_path.stem}_results.txt"

    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        return
    if not video_path.exists():
        print(f"❌ Error: Video not found at {video_path}")
        return

    print(f"✅ Loading model from: {model_path}")
    model = YOLO(model_path)

    print(f"⏳ Processing video: {video_path.name}")
    results_generator = model.track(
        source=str(video_path),
        tracker="bytetrack.yaml",
        persist=True,
        save=True,
        project=str(output_dir),
        name=f"{video_path.stem}_video_output",
        conf=0.3,
        stream=True,
        verbose=False
    )

    with open(txt_output_path, "w") as f:
        for frame_number, results in enumerate(results_generator, 1):
            if results.boxes.id is None:
                continue

            boxes_xywh = results.boxes.xywh.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist()
            confidences = results.boxes.conf.cpu().numpy()

            for box, track_id, conf in zip(boxes_xywh, track_ids, confidences):
                x_center, y_center, width, height = box
                x_tl = x_center - width / 2
                y_tl = y_center - height / 2
                f.write(f"{frame_number},{track_id},{x_tl:.2f},{y_tl:.2f},{width:.2f},{height:.2f},{conf:.2f},-1,-1,-1\n")

    print(f"✅ Tracking complete. Data saved to: {txt_output_path}")


if __name__ == "__main__":
    # --- Setup Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run object tracking on a video.")
    parser.add_argument("--model", default="models/best.pt", help="Path to the YOLO model.")
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--output-dir", default="phase3_outputs", help="Directory to save the results.")
    
    args = parser.parse_args()

    run_tracker(args.model, args.video, args.output_dir)