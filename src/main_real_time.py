import cv2
from ultralytics import YOLO
from pathlib import Path
import argparse
import numpy as np
import torch
from PIL import Image
from classifier.models import load_species_classifier
from classifier.config import INFERENCE_TRANSFORMS



# --- Main Simulation Pipeline Function ---
def run_pipeline(yolo_model_path, cnn_model_path, input_source, output_dir):
    """
    Executes the full detection, tracking, and classification pipeline.
    """
    # --- 1. SETUP ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    is_real_time = not Path(input_source).exists()

    # --- 2. LOAD MODELS ---
    print(f"✅ Loading YOLO 'fish' tracker model: {yolo_model_path}")
    yolo_model = YOLO(yolo_model_path)

    # Load the CNN model AND the species list from the checkpoint file
    print(f"✅ Loading species classifier model: {cnn_model_path}")
    species_classifier, species_list = load_species_classifier(cnn_model_path, device)

    # --- 3. SETUP VIDEO I/O ---
    source = int(input_source) if is_real_time else input_source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source.")
        return
        
    video_writer = None
    if not is_real_time:
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_output_path = output_dir / f"{Path(input_source).stem}_classified.mp4"
        video_writer = cv2.VideoWriter(str(video_output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    np.random.seed(42)
    colors = np.random.randint(0, 256, size=(100, 3), dtype=np.uint8)
    species_map = {}

    print("\n🚀 Starting pipeline... Press 'q' in the video window to quit.")
    
    # --- 4. PROCESSING LOOP ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model.track(frame, persist=True, conf=0.3, verbose=False)

        if results[0].boxes.id is not None:
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes_xyxy, track_ids):
                if track_id not in species_map:
                    x1, y1, x2, y2 = map(int, box)
                    cropped_fish = frame[y1:y2, x1:x2]
                    
                    if cropped_fish.size > 0:
                        pil_image = Image.fromarray(cv2.cvtColor(cropped_fish, cv2.COLOR_BGR2RGB))
                        input_tensor = INFERENCE_TRANSFORMS(pil_image).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            outputs = species_classifier(input_tensor)
                            _, predicted_idx = torch.max(outputs, 1)
                            species_name = species_list[predicted_idx.item()]
                        
                        species_map[track_id] = species_name
                        print(f"   -> New fish! ID {track_id} classified as '{species_name}'")

                species_name = species_map.get(track_id, "fish")
                label_text = f"id:{track_id} {species_name}"
                
                x1, y1, x2, y2 = map(int, box)
                color = tuple(colors[track_id % 100].tolist())
                
                (w_text, h_text), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w_text, y1), color, -1)
                cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Always show the window
        cv2.imshow("Fish Pipeline", frame)

        # If it's not a real-time feed, also write to the file
        if not is_real_time:
            video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 5. CLEANUP ---
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    print("✅ Pipeline stopped.")
    if not is_real_time:
        print(f"🎥 Final video saved to: {video_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Fish Tracking and Classification Pipeline.")
    
    parser.add_argument("--source", required=True, help="Path to the input video file OR camera ID (e.g., '0').")
    parser.add_argument("--yolo-model", default="models/fish_detection.pt", help="Path to the single-class 'fish' YOLO tracker model.")
    parser.add_argument("--cnn-model", default="models/fish_classification.pth", help="Path to your CNN species classifier checkpoint.")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save the results.")
    
    args = parser.parse_args()

    # The species list is now loaded dynamically from the model, so it's no longer needed here.
    run_pipeline(args.yolo_model, args.cnn_model, args.source, args.output_dir)