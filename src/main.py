# main_pipeline.py
# A complete script for detecting, tracking, and classifying fish species in a video.

import cv2
from ultralytics import YOLO
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image

# --- Placeholder for your CNN Species Classifier ---
def get_species_classifier(model_path, num_species, device):
    """
    Loads your pre-trained CNN species classifier.
    NOTE: You MUST replace this with your actual model architecture.
    """
    # Example: Using a pre-trained ResNet model as a placeholder
    import torchvision.models as models
    
    # Load a pre-trained model and reset the final fully connected layer.
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_species) # Adjust to your number of species
    
    # Load your trained weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"⚠️  Warning: Could not load species model weights from {model_path}. Using a randomly initialized model. Error: {e}")
        
    model.to(device)
    model.eval() # Set the model to evaluation mode
    return model

# Define the image transformations your CNN expects
cnn_image_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # Example size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# --- Main Pipeline Function ---
def run_pipeline(yolo_model_path, cnn_model_path, video_path, output_dir, species_list, min_track_duration=5):
    """
    Executes the full detection, tracking, and classification pipeline.
    """
    # --- 1. SETUP ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    video_path = Path(video_path)
    video_output_path = output_dir / f"{video_path.stem}_classified.mp4"

    # --- 2. LOAD MODELS ---
    print(f"✅ Loading YOLO 'fish' tracker model: {yolo_model_path}")
    yolo_model = YOLO(yolo_model_path)

    print(f"✅ Loading CNN species classifier: {cnn_model_path}")
    species_classifier = get_species_classifier(cnn_model_path, len(species_list), device)

    # --- 3. FIRST PASS: DETECT & TRACK ALL OBJECTS ---
    print(f"⏳ Pass 1/2: Collecting all raw tracking data...")
    all_tracks_data = []
    cap = cv2.VideoCapture(str(video_path))
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        # Use the high-level, stable model.track() function
        results = yolo_model.track(frame, persist=True, conf=0.1, verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = box
                all_tracks_data.append([frame_number, track_id, x1, y1, x2, y2])
    cap.release()
    print(f"✅ Pass 1 complete. Found {len(all_tracks_data)} raw detections.")

    if not all_tracks_data:
        print("No objects tracked. Exiting.")
        return

    # --- 4. ANALYZE TRACKS & CLASSIFY SPECIES ---
    print(f"⏳ Analyzing tracks and classifying species...")
    tracks_df = pd.DataFrame(all_tracks_data, columns=['frame_id', 'track_id', 'x1', 'y1', 'x2', 'y2'])
    
    track_durations = tracks_df.groupby('track_id').size()
    stable_track_ids = track_durations[track_durations >= min_track_duration].index
    
    species_map = {} # Dictionary to store: {track_id: "species_name"}

    cap = cv2.VideoCapture(str(video_path)) # Re-open video for cropping
    frame_number = 0
    classified_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        
        # Find tracks present in this frame
        frame_tracks = tracks_df[tracks_df['frame_id'] == frame_number]
        
        for _, track in frame_tracks.iterrows():
            track_id = int(track['track_id'])
            
            # Classify this track ID only if it's stable and hasn't been classified yet
            if track_id in stable_track_ids and track_id not in classified_ids:
                x1, y1, x2, y2 = map(int, [track['x1'], track['y1'], track['x2'], track['y2']])
                
                # Crop the fish image
                cropped_fish = frame[y1:y2, x1:x2]
                
                if cropped_fish.size > 0:
                    # Convert to PIL Image and apply transformations
                    pil_image = Image.fromarray(cv2.cvtColor(cropped_fish, cv2.COLOR_BGR2RGB))
                    input_tensor = cnn_image_transforms(pil_image).unsqueeze(0).to(device)
                    
                    # Get species prediction from the CNN
                    with torch.no_grad():
                        outputs = species_classifier(input_tensor)
                        _, predicted_idx = torch.max(outputs, 1)
                        species_name = species_list[predicted_idx.item()]
                        
                        # Store the species for this track ID
                        species_map[track_id] = species_name
                        classified_ids.add(track_id)
                        print(f"   -> Classified Track ID {track_id} as '{species_name}'")

    cap.release()
    print(f"✅ Classification complete. Identified {len(species_map)} unique fish.")

    # --- 5. SECOND PASS: DRAW FINAL VIDEO ---
    print(f"⏳ Pass 2/2: Drawing final annotated video...")
    cap = cv2.VideoCapture(str(video_path))
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(str(video_output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    
    np.random.seed(42)
    colors = np.random.randint(0, 256, size=(max(tracks_df['track_id'].max() + 1, 100), 3), dtype=np.uint8)
    
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        
        frame_tracks = tracks_df[tracks_df['frame_id'] == frame_number]
        
        for _, track in frame_tracks.iterrows():
            track_id = int(track['track_id'])
            
            # Only draw boxes for tracks that have been classified
            if track_id in species_map:
                x1, y1, x2, y2 = map(int, [track['x1'], track['y1'], track['x2'], track['y2']])
                
                # Get the persistent species name
                species_name = species_map[track_id]
                label_text = f"id:{track_id} {species_name}"
                color = tuple(colors[track_id % 100].tolist())
                
                (w_text, h_text), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w_text, y1), color, -1)
                cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
        video_writer.write(frame)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print("-" * 30)
    print(f"✅ Pipeline complete!")
    print(f"🎥 Final video saved to: {video_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Fish Tracking and Classification Pipeline.")
    
    # --- REQUIRED ARGUMENTS ---
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    
    # --- MODEL PATHS ---
    parser.add_argument("--yolo-model", default="models/fish_detection.pt", help="Path to the single-class 'fish' YOLO tracker model.")
    parser.add_argument("--cnn-model", default="models/fish_classification.pth", help="Path to your CNN species classifier model.")
    
    # --- OUTPUT & OTHER SETTINGS ---
    parser.add_argument("--output-dir", default="outputs", help="Directory to save the results.")
    parser.add_argument("--min-duration", type=int, default=5, help="Minimum frames a track must exist to be classified.")

    args = parser.parse_args()

    # --- DEFINE YOUR SPECIES ---
    # IMPORTANT: The order of this list MUST match the output order of your CNN model.
    # e.g., if your CNN outputs '0' for Tuna, '1' for Salmon, etc.
    SPECIES_LIST = [
        "Pristipomoides Auricilla",
        "Pristipomoides Zonatus",
        "Species C", 
        # ... add all your species here
    ]

    run_pipeline(args.yolo_model, args.cnn_model, args.video, args.output_dir, SPECIES_LIST, args.min_duration)
