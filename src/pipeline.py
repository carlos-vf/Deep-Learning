import cv2
from pathlib import Path
import numpy as np

# --- STANDARD REAL-TIME / SINGLE-PASS ---
def run_standard_pipeline(yolo_model, input_source, output_dir, tracker_config_path):
    """
    Executes a real-time pipeline using a single multi-class YOLO model.
    This mode processes frames as they come, providing immediate visual feedback.
    If the input is a video file, it saves both an annotated video and a .txt data file.
    """
    # --- SETUP ---
    # Determine if the source is a live camera feed or a pre-recorded video file.
    is_real_time = not Path(input_source).exists()
    
    video_writer = None
    txt_file = None
    
    # If processing a video file, set up the output paths and file writers.
    if not is_real_time:
        video_path = Path(input_source)
        video_output_path = Path(output_dir) / f"{video_path.stem}_standard_classified.mp4"
        txt_output_path = Path(output_dir) / f"{video_path.stem}_standard_results.txt"
        txt_file = open(txt_output_path, "w")

        # Get video properties (width, height, fps) to correctly initialize the VideoWriter.
        cap_meta = cv2.VideoCapture(input_source)
        w, h, fps = (int(cap_meta.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter(str(video_output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        cap_meta.release()

    # --- INITIALIZE BUFFERS ---
    # Generate a consistent color palette for different track IDs.
    np.random.seed(42)
    colors = np.random.randint(0, 256, size=(100, 3), dtype=np.uint8)
    # This dictionary will store the most confident species prediction for each track ID.
    # Format: {track_id: ("species_name", confidence_score)}
    species_map = {}

    print(f"\n🚀 Starting Standard Real-Time Pipeline (Config: {tracker_config_path})... Press 'q' to quit.")
    
    # --- MAIN PROCESSING LOOP ---
    # Use model.track() with stream=True for efficient, memory-safe video processing.
    results_generator = yolo_model.track(source=input_source, stream=True, persist=True, conf=0.3, tracker=tracker_config_path, verbose=False)

    for frame_number, results in enumerate(results_generator, 1):
        frame = results.orig_img

        if results.boxes.id is not None:
            # Extract tracking data for the current frame.
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist()
            confs = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.int().cpu().tolist()

            for box, track_id, conf, cls_id in zip(boxes_xyxy, track_ids, confs, class_ids):
                x1, y1, x2, y2 = map(int, box)
                
                # If processing a video, save the raw tracking data to the text file.
                if txt_file:
                    width = x2 - x1
                    height = y2 - y1
                    txt_file.write(f"{frame_number},{track_id},{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},{conf:.2f},-1,-1,-1\n")

                # --- Confidence-Based Classification Logic ---
                # Get the species prediction from the current frame's detection.
                new_species = yolo_model.names[cls_id]
                new_conf = conf

                # Get the best confidence score stored so far for this track ID.
                current_best_conf = species_map.get(track_id, ("Unknown", 0.0))[1]
                
                # Update the species label only if the new detection is more confident.
                if new_conf > current_best_conf:
                    species_map[track_id] = (new_species, new_conf)

                # --- Drawing Logic ---
                # Get the most confident species name seen so far for this track.
                species_name, _ = species_map.get(track_id, ("(tracking...)", 0.0))
                label_text = f"id:{track_id} {species_name}"
                
                color = tuple(colors[track_id % 100].tolist())
                (w_text, h_text), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w_text, y1), color, -1)
                cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Display the annotated frame in a window.
        cv2.imshow("Fish Pipeline", frame)
        # If processing a video, write the frame to the output file.
        if video_writer:
            video_writer.write(frame)
        # Allow quitting by pressing 'q'.
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


# --- BUFFERED REAL-TIME (SINGLE-PASS, DELAYED) ---
def run_buffered_pipeline(yolo_model, input_source, output_dir, tracker_config_path, min_track_duration):
    """
    Executes a single-pass pipeline with a buffer to validate tracks. This mode
    only displays and records tracks after they have been stable for a minimum
    number of frames, reducing noise from fleeting detections.
    """
    # --- SETUP ---
    is_real_time = not Path(input_source).exists()
    
    video_writer = None
    txt_file = None

    if not is_real_time:
        video_path = Path(input_source)
        video_output_path = Path(output_dir) / f"{video_path.stem}_buffered_classified.mp4"
        txt_output_path = Path(output_dir) / f"{video_path.stem}_buffered_results.txt"
        txt_file = open(txt_output_path, "w")

        cap_meta = cv2.VideoCapture(input_source)
        w, h, fps = (int(cap_meta.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter(str(video_output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        cap_meta.release()

    # --- INITIALIZE BUFFERS ---
    np.random.seed(42)
    colors = np.random.randint(0, 256, size=(100, 3), dtype=np.uint8)
    # Stores the most confident species prediction for each track ID.
    species_map = {} 
    # Stores the number of consecutive frames each track has been seen.
    track_history = {}

    print(f"\n🚀 Starting Buffered Pipeline... Press 'q' to quit.")
    
    # --- MAIN PROCESSING LOOP ---
    results_generator = yolo_model.track(source=input_source, stream=True, persist=True, conf=0.1, tracker=tracker_config_path, verbose=False)

    for frame_number, results in enumerate(results_generator, 1):
        frame = results.orig_img

        if results.boxes.id is not None:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist()
            confs = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.int().cpu().tolist()

            # Update the history for each currently visible track.
            for track_id in track_ids:
                track_history[track_id] = track_history.get(track_id, 0) + 1

            for box, track_id, conf, cls_id in zip(boxes_xyxy, track_ids, confs, class_ids):
                # --- Temporal Filtering Logic ---
                # Only process and draw tracks that have been stable for the minimum duration.
                if track_history.get(track_id, 0) >= min_track_duration:
                    new_species = yolo_model.names[cls_id]
                    new_conf = conf

                    current_best_conf = species_map.get(track_id, ("Unknown", 0.0))[1]
                    if new_conf > current_best_conf:
                        species_map[track_id] = (new_species, new_conf)

                    species_name, _ = species_map.get(track_id, ("(tracking...)", 0.0))
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