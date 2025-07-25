import argparse
import torch
from ultralytics import YOLO
from pathlib import Path 
from pipeline import run_standard_pipeline, run_buffered_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Fish Tracking and Classification Pipeline.")
    
    parser.add_argument("--source", required=True, help="Path to the input video file OR camera ID (e.g., '0').")
    parser.add_argument("--yolo-model", default="models/deepfish_multi_m.pt", help="Path to the single-class 'deepfish_multi' YOLO tracker model.")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save the results.")
    parser.add_argument("--tracker-config", default="src/tracker/bytetrack.yaml", help="Path to the tracker configuration file.")
    parser.add_argument("--mode", default="standard", choices=["standard", "buffered"],
                        help="Processing mode: 'standard' or 'buffered'.")
    parser.add_argument("--min-duration", type=int, default=2, 
                        help="Minimum frames for a track to be considered stable (for buffered mode).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(output_dir / "logs" / f"{args.mode}").mkdir(parents=True, exist_ok=True)
    Path(output_dir / "videos" / f"{args.mode}").mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directory is set to: {output_dir.resolve()}")

    # Define the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load models once
    yolo_model = YOLO(args.yolo_model).to(device)
    yolo_model.info() 

    # Logic to call the correct pipeline based on the selected mode
    if args.mode == "standard":
        run_standard_pipeline(yolo_model, args.source, args.output_dir, args.tracker_config)
    elif args.mode == "buffered":
        run_buffered_pipeline(yolo_model, args.source, args.output_dir, args.tracker_config, args.min_duration)
