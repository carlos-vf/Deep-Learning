import argparse
import torch
from ultralytics import YOLO
from pipeline import run_real_time_pipeline, run_buffered_pipeline
from classifier.models import load_species_classifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Fish Tracking and Classification Pipeline.")
    
    parser.add_argument("--source", required=True, help="Path to the input video file OR camera ID (e.g., '0').")
    parser.add_argument("--yolo-model", default="models/fish_detection.pt", help="Path to the single-class 'fish' YOLO tracker model.")
    parser.add_argument("--cnn-model", default="models/fish_classification.pth", help="Path to your CNN species classifier checkpoint.")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save the results.")
    parser.add_argument("--tracker-config", default="src/tracker/bytetrack.yaml", help="Path to the tracker configuration file.")
    parser.add_argument("--mode", default="standard", choices=["standard", "buffered"],
                        help="Processing mode: 'realtime' or 'buffered'.")
    parser.add_argument("--min-duration", type=int, default=5, 
                        help="Minimum frames for a track to be considered stable (for buffered mode).")
    args = parser.parse_args()

    # Define the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load models once
    yolo_model = YOLO(args.yolo_model).to(device)
    cnn_model, species_list = load_species_classifier(args.cnn_model, device)

    # Logic to call the correct pipeline based on the selected mode
    if args.mode == "standard":
        run_real_time_pipeline(yolo_model, cnn_model, species_list, args.source, args.output_dir, args.tracker_config, device)
    elif args.mode == "buffered":
        run_buffered_pipeline(yolo_model, cnn_model, species_list, args.source, args.output_dir, args.tracker_config, args.min_duration, device)
