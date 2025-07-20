import cv2
from pathlib import Path
import numpy as np



import cv2
from pathlib import Path
import numpy as np

# --- STANDARD REAL-TIME / SINGLE-PASS ---
def run_standard_pipeline(yolo_model, input_source, output_dir, tracker_config_path):
    """
    Executes a real-time pipeline that adapts its labeling strategy based on the
    provided YOLO model (single-class vs. multi-class).
    """
    # --- SETUP ---
    is_real_time = not Path(input_source).exists()
    is_multiclass = len(yolo_model.names) > 1
    
    video_writer = None
    txt_file = None
    
    if not is_real_time:
        video_path = Path(input_source)
        video_output_path = Path(output_dir) / "videos/standard" / f"{video_path.stem}_standard_classified.mp4"
        txt_output_path = Path(output_dir) / "logs/standard" / f"{video_path.stem}_standard_results.txt"
        txt_file = open(txt_output_path, "w")

        cap_meta = cv2.VideoCapture(input_source)
        w, h, fps = (int(cap_meta.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter(str(video_output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        cap_meta.release()

    np.random.seed(42)
    colors = np.random.randint(0, 256, size=(100, 3), dtype=np.uint8)
    # This buffer is only needed for multi-class models to stabilize labels.
    species_map = {}

    print(f"\n🚀 Starting Standard Real-Time Pipeline (Config: {tracker_config_path})... Press 'q' to quit.")
    
    # --- MAIN PROCESSING LOOP ---
    results_generator = yolo_model.track(source=input_source, stream=True, persist=True, conf=0.3, tracker=tracker_config_path, verbose=False)

    for frame_number, results in enumerate(results_generator, 1):
        frame = results.orig_img

        if results.boxes.id is not None:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist()
            confs = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.int().cpu().tolist()

            for box, track_id, conf, cls_id in zip(boxes_xyxy, track_ids, confs, class_ids):
                x1, y1, x2, y2 = map(int, box)
                
                if txt_file:
                    width = x2 - x1
                    height = y2 - y1
                    txt_file.write(f"{frame_number},{track_id},{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},{conf:.2f},-1,-1,-1\n")

                if is_multiclass:
                    # For multi-class models, use the confidence-based update strategy
                    new_species = yolo_model.names[cls_id]
                    current_best_conf = species_map.get(track_id, ("Unknown", 0.0))[1]
                    if conf > current_best_conf:
                        species_map[track_id] = (new_species, conf)
                    species_name, _ = species_map.get(track_id, ("(tracking...)", 0.0))
                else:
                    # For single-class models, just use the "fish" label
                    species_name = "fish"
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

    if video_writer:
        video_writer.release()
        print(f"🎥 Final video saved to: {video_output_path}")
    if txt_file:
        txt_file.close()
        print(f"📄 Tracking data saved to: {txt_output_path}")
    cv2.destroyAllWindows()
    print("✅ Pipeline stopped.")


# --- BUFFERED REAL-TIME (SINGLE-PASS, DELAYED) ---
def run_buffered_pipeline(yolo_model, input_source, output_dir, tracker_config_path, min_track_duration):
    """
    Executes a single-pass pipeline with a buffer to validate tracks.
    Adapts labeling based on whether the model is single-class or multi-class.
    """
    is_real_time = not Path(input_source).exists()
    is_multiclass = len(yolo_model.names) > 1
    
    video_writer = None
    txt_file = None

    if not is_real_time:
        video_path = Path(input_source)
        video_output_path = Path(output_dir) / "videos/buffered" / f"{video_path.stem}_buffered_classified.mp4"
        txt_output_path = Path(output_dir) / "logs/buffered" / f"{video_path.stem}_buffered_results.txt"
        txt_file = open(txt_output_path, "w")

        cap_meta = cv2.VideoCapture(input_source)
        w, h, fps = (int(cap_meta.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter(str(video_output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        cap_meta.release()

    np.random.seed(42)
    colors = np.random.randint(0, 256, size=(100, 3), dtype=np.uint8)
    
    species_map = {} 
    track_history = {}

    print(f"\n🚀 Starting Buffered Pipeline... Press 'q' to quit.")
    
    results_generator = yolo_model.track(source=input_source, stream=True, persist=True, conf=0.1, tracker=tracker_config_path, verbose=False)

    for frame_number, results in enumerate(results_generator, 1):
        frame = results.orig_img

        if results.boxes.id is not None:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist()
            confs = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.int().cpu().tolist()

            for track_id in track_ids:
                track_history[track_id] = track_history.get(track_id, 0) + 1

            for box, track_id, conf, cls_id in zip(boxes_xyxy, track_ids, confs, class_ids):
                if track_history.get(track_id, 0) >= min_track_duration:
                    
                    if is_multiclass:
                        new_species = yolo_model.names[cls_id]
                        current_best_conf = species_map.get(track_id, ("Unknown", 0.0))[1]
                        if conf > current_best_conf:
                            species_map[track_id] = (new_species, conf)
                        species_name, _ = species_map.get(track_id, ("(tracking...)", 0.0))
                    else:
                        species_name = "fish"

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

    # --- CLEANUP ---
    if video_writer:
        video_writer.release()
        print(f"🎥 Final video saved to: {video_output_path}")
    if txt_file:
        txt_file.close()
        print(f"📄 Tracking data saved to: {txt_output_path}")
    cv2.destroyAllWindows()
    print("✅ Pipeline stopped.")