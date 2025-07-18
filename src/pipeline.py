import cv2
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from classifier.config import INFERENCE_TRANSFORMS

# --- REAL-TIME / SINGLE-PASS ---
def run_real_time_pipeline(yolo_model, species_classifier, species_list, input_source, output_dir, tracker_config_path, device):
    """
    Executes the real-time, single-pass pipeline.
    If input is a video file, it saves both an annotated video and a .txt data file.
    """
    is_real_time = not Path(input_source).exists()
    
    cap = cv2.VideoCapture(int(input_source) if is_real_time else input_source)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source '{input_source}'.")
        return
        
    video_writer = None
    txt_file = None
    
    if not is_real_time:
        video_path = Path(input_source)
        video_output_path = Path(output_dir) / f"{video_path.stem}_realtime_classified.mp4"
        txt_output_path = Path(output_dir) / f"{video_path.stem}_realtime_results.txt"
        
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter(str(video_output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        txt_file = open(txt_output_path, "w")

    np.random.seed(42)
    colors = np.random.randint(0, 256, size=(100, 3), dtype=np.uint8)
    
    species_map = {}

    print(f"\n🚀 Starting Real-Time Pipeline (Config: {tracker_config_path})... Press 'q' to quit.")
    
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        results = yolo_model.track(frame, persist=True, conf=0.3, tracker=tracker_config_path, verbose=False)

        if results[0].boxes.id is not None:
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, track_id, conf in zip(boxes_xyxy, track_ids, confs):
                x1, y1, x2, y2 = map(int, box)
                
                # --- Save data to text file ---
                if txt_file:
                    width = x2 - x1
                    height = y2 - y1
                    txt_file.write(f"{frame_number},{track_id},{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},{conf:.2f},-1,-1,-1\n")

                # Classify the object on its first appearance or if a better view might be available
                cropped_fish = frame[y1:y2, x1:x2]
                if cropped_fish.size > 0:
                    pil_image = Image.fromarray(cv2.cvtColor(cropped_fish, cv2.COLOR_BGR2RGB))
                    input_tensor = INFERENCE_TRANSFORMS(pil_image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = species_classifier(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        new_conf, predicted_idx = torch.max(probabilities, 1)
                        
                        new_species = species_list[predicted_idx.item()]
                        new_conf = new_conf.item()

                        # Update only if the new classification is more confident
                        current_best_conf = species_map.get(track_id, ("Unknown", 0.0))[1]
                        if new_conf > current_best_conf:
                            species_map[track_id] = (new_species, new_conf)
                
                # --- Drawing Logic ---
                species_name, species_conf = species_map.get(track_id, ("(classifying...)", 0.0))
                label_text = f"id:{track_id} {species_name}"
                
                color = tuple(colors[track_id % 100].tolist())
                (w_text, h_text), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w_text, y1), color, -1)
                cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.imshow("Fish Pipeline", frame)
        if video_writer:
            video_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if video_writer:
        video_writer.release()
        print(f"🎥 Final video saved to: {video_output_path}")
    if txt_file:
        txt_file.close()
        print(f"📄 Tracking data saved to: {txt_output_path}")
    cv2.destroyAllWindows()
    print("✅ Pipeline stopped.")



# --- BUFFERED REAL-TIME (SINGLE-PASS, DELAYED) ---
def run_buffered_pipeline(yolo_model, species_classifier, species_list, input_source, output_dir, tracker_config_path, min_track_duration, classify_interval, device):
    """
    Executes a single-pass pipeline with a buffer to validate tracks.
    If input is a video file, it saves both an annotated video and a .txt data file.
    """
    is_real_time = not Path(input_source).exists()
    
    cap = cv2.VideoCapture(int(input_source) if is_real_time else input_source)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source '{input_source}'.")
        return
        
    video_writer = None
    txt_file = None

    if not is_real_time:
        video_path = Path(input_source)
        # Setup paths for both video and text file outputs
        video_output_path = Path(output_dir) / f"{video_path.stem}_buffered_classified.mp4"
        txt_output_path = Path(output_dir) / f"{video_path.stem}_buffered_results.txt"
        
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter(str(video_output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        txt_file = open(txt_output_path, "w")

    np.random.seed(42)
    colors = np.random.randint(0, 256, size=(100, 3), dtype=np.uint8)
    # --- Data Buffers ---
    species_map = {} 
    track_history = {}
    last_classification_frame = {}

    print(f"\n🚀 Starting Buffered Pipeline (Classify Interval: {classify_interval} frames per fish)... Press 'q' to quit.")
    
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        results = yolo_model.track(frame, persist=True, conf=0.1, tracker=tracker_config_path, verbose=False)

        if results[0].boxes.id is not None:
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().numpy()

            for track_id in track_ids:
                track_history[track_id] = track_history.get(track_id, 0) + 1

            for box, track_id, conf in zip(boxes_xyxy, track_ids, confs):
                if track_history.get(track_id, 0) >= min_track_duration:
                    
                    # Check if enough frames have passed since the last classification FOR THIS SPECIFIC TRACK
                    if frame_number - last_classification_frame.get(track_id, -classify_interval) >= classify_interval:
                        x1, y1, x2, y2 = map(int, box)
                        cropped_fish = frame[y1:y2, x1:x2]
                        
                        if cropped_fish.size > 0:
                            pil_image = Image.fromarray(cv2.cvtColor(cropped_fish, cv2.COLOR_BGR2RGB))
                            input_tensor = INFERENCE_TRANSFORMS(pil_image).unsqueeze(0).to(device)
                            
                            with torch.no_grad():
                                outputs = species_classifier(input_tensor)
                                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                                new_conf, predicted_idx = torch.max(probabilities, 1)
                                
                                new_species = species_list[predicted_idx.item()]
                                new_conf = new_conf.item()

                                # Update only if the new classification is more confident
                                current_best_conf = species_map.get(track_id, ("Unknown", 0.0))[1]
                                if new_conf > current_best_conf:
                                    species_map[track_id] = (new_species, new_conf)
                                    #print(f"   -> Updated ID {track_id} to '{new_species}' (Conf: {new_conf:.2f})")
                        
                        # Update the last classification frame for this track ID
                        last_classification_frame[track_id] = frame_number

                    # --- Drawing Logic ---
                    species_name, _ = species_map.get(track_id, ("(classifying...)", 0.0))
                    label_text = f"id:{track_id} {species_name}"
                    
                    x1, y1, x2, y2 = map(int, box)
                    color = tuple(colors[track_id % 100].tolist())
                    
                    (w_text, h_text), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + w_text, y1), color, -1)
                    cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    if txt_file:
                        width = x2 - x1
                        height = y2 - y1
                        txt_file.write(f"{frame_number},{track_id},{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},{conf:.2f},-1,-1,-1\n")

        cv2.imshow("Fish Pipeline", frame)
        if video_writer:
            video_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if video_writer:
        video_writer.release()
        print(f"🎥 Final video saved to: {video_output_path}")
    if txt_file:
        txt_file.close()
        print(f"📄 Tracking data saved to: {txt_output_path}")
    cv2.destroyAllWindows()
    print("✅ Pipeline stopped.")
