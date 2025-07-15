# simulate_real_time.py
# Simulates the real-time pipeline using a pre-recorded video file.

import cv2
from ultralytics import YOLO
from pathlib import Path
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# --- Placeholder for your CNN Species Classifier ---
# You MUST replace this with your actual model definition and weights.
def get_species_classifier(model_path, num_species, device):
    """
    Loads your pre-trained CNN species classifier.
    NOTE: This is a placeholder. You must adapt it to your model's architecture.
    """
    import torchvision.models as models
    model = models.resnet18(weights=None) # Load architecture without pre-trained weights
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_species)
    
    try:
        # Load your trained weights onto the correct device
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Successfully loaded species model weights from {model_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not load species model weights. Using a randomly initialized model. Error: {e}")
        
    model.to(device)
    model.eval()
    return model

# Define the image transformations your CNN expects
# NOTE: These MUST match the transformations used during your CNN training.
cnn_image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Main Simulation Pipeline Function ---

def run_simulation_pipeline(yolo_model_path, cnn_model_path, species_list, video_path, output_dir):
    """
    Executes the full detection, tracking, and classification pipeline on a video file,
    simulating the real-time, single-pass workflow.
    """
    # --- 1. SETUP ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    video_output_path = output_dir / f"{video_path.stem}_simulation_output.mp4"

    # --- 2. LOAD MODELS ---
    print(f"✅ Loading YOLO 'fish' tracker model: {yolo_model_path}")
    yolo_model = YOLO(yolo_model_path)

    print(f"✅ Loading CNN species classifier: {cnn_model_path}")
    species_classifier = get_species_classifier(cnn_model_path, len(species_list), device)

    # --- 3. SETUP VIDEO I/O ---
    print(f"⏳ Opening video file: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file.")
        return
        
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(str(video_output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    np.random.seed(42)
    colors = np.random.randint(0, 256, size=(100, 3), dtype=np.uint8)
    
    # This dictionary will store the classified species for each track ID
    species_map = {}

    print("\n🚀 Starting simulation... Press 'q' in the video window to quit early.")
    
    # --- 4. REAL-TIME SIMULATION LOOP ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Use the stable, high-level track function
        results = yolo_model.track(frame, persist=True, conf=0.3, verbose=False)

        if results[0].boxes.id is not None:
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes_xyxy, track_ids):
                # --- "CLASSIFY-ON-FIRST-SEEN" LOGIC ---
                if track_id not in species_map:
                    x1, y1, x2, y2 = map(int, box)
                    cropped_fish = frame[y1:y2, x1:x2]
                    
                    species_name = "Unknown"
                    if cropped_fish.size > 0:
                        pil_image = Image.fromarray(cv2.cvtColor(cropped_fish, cv2.COLOR_BGR2RGB))
                        input_tensor = cnn_image_transforms(pil_image).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            outputs = species_classifier(input_tensor)
                            _, predicted_idx = torch.max(outputs, 1)
                            species_name = species_list[predicted_idx.item()]
                    
                    species_map[track_id] = species_name
                    print(f"   -> New fish detected! ID {track_id} classified as '{species_name}'")

                # --- DRAW ANNOTATIONS ---
                species_name = species_map.get(track_id, "fish")
                label_text = f"id:{track_id} {species_name}"
                
                x1, y1, x2, y2 = map(int, box)
                color = tuple(colors[track_id % 100].tolist())
                
                (w_text, h_text), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w_text, y1), color, -1)
                cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Display the frame to simulate a real-time feed
        cv2.imshow("Real-Time Simulation", frame)
        
        # Write the annotated frame to the output video file
        video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 5. CLEANUP ---
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print("✅ Pipeline simulation complete.")
    print(f"🎥 Final video saved to: {video_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Fish Tracking and Classification Pipeline Simulation.")
    
    # --- REQUIRED ARGUMENTS ---
    parser.add_argument("--video", required=True, help="Path to the input video file for simulation.")
    
    # --- MODEL PATHS ---
    parser.add_argument("--yolo-model", default="models/fish_detection.pt", help="Path to the single-class 'fish' YOLO tracker model.")
    parser.add_argument("--cnn-model", default="models/fish_classification.pth", help="Path to your CNN species classifier model.")
    
    # --- OUTPUT ---
    parser.add_argument("--output-dir", default="pipeline_output", help="Directory to save the results.")
    
    args = parser.parse_args()

    # --- DEFINE YOUR SPECIES ---
    # IMPORTANT: The order of this list MUST match the output order of your CNN model.
    SPECIES_LIST = [
        "Pristipomoides Auricilla",
        "Pristipomoides Zonatus",
        "Species C", 
        # ... add all your species here
    ]

    run_simulation_pipeline(args.yolo_model, args.cnn_model, SPECIES_LIST, args.video, args.output_dir)
