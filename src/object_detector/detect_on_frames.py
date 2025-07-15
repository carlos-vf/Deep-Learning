# src/detect_on_frames.py

from ultralytics import YOLO
from pathlib import Path

def detect_on_folder(model_path, frames_folder_path):
    """
    Runs detection on a folder of images and saves the annotated results.

    Args:
        model_path (str): Path to the trained YOLOv8 model (.pt file).
        frames_folder_path (str): Path to the folder containing image frames.
    """
    model_path = Path(model_path)
    frames_folder_path = Path(frames_folder_path)

    # Check if paths are valid
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        return
    if not frames_folder_path.exists() or not frames_folder_path.is_dir():
        print(f"❌ Error: Frames folder not found at {frames_folder_path}")
        return

    # Load your custom-trained model
    print(f"✅ Loading model from: {model_path}")
    model = YOLO(model_path)

    # Run prediction on the folder
    # The library will automatically find all images in the folder
    print(f"⏳ Running detection on folder: {frames_folder_path.name}")
    results = model.predict(
        source=str(frames_folder_path),
        save=True,  # Save the output images with detections
        conf=0.25   # Confidence threshold for detections
    )

    # The results are saved automatically, but this loop is useful to see progress
    for i, result in enumerate(results):
        pass # The 'save=True' argument handles the saving

    print(f"\n✅ Detection complete. Results saved in the 'runs/detect/' directory.")


if __name__ == '__main__':
    # --- Configuration ---
    # Adjust these paths to match your project structure
    MODEL_PATH = "models/best.pt"
    FRAMES_FOLDER_PATH = "data/FISHTRAC/V1_Leleiwi_26June19_17" # Example folder

    detect_on_folder(MODEL_PATH, FRAMES_FOLDER_PATH)