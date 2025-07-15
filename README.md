# Underwater Fish Detection, Tracking, and Classification



## Project Overview

This project implements a complete computer vision pipeline to process underwater video footage. The system is designed to perform three core tasks:

1.  **Detection**: Identify and locate fish within each frame of a video.
2.  **Tracking**: Assign a unique, persistent ID to each detected fish and follow it across multiple frames.
3.  **Classification**: For each tracked fish, identify its specific species using a dedicated classification model.

The pipeline is built to be flexible, offering multiple processing modes ranging from fast, real-time analysis for live camera feeds to more computationally intensive, high-accuracy offline processing for pre-recorded videos.

---

## Project Structure

The project is organized into the following directories to ensure a clean and scalable workflow:

```
Deep-Learning/
├── data/
│   └── ... (Contains raw datasets like F4K, Brackish, DeepFish)
├── models/
│   ├── fish_detection.pt      # Single-class YOLO model for tracking
│   └── fish_classification.pth # Custom CNN model for species
├── outputs/
│   └── ... (Generated videos and data files are saved here)
├── src/
│   ├── main.py              # The main entry point for the application
│   ├── pipeline.py             
│   ├── classifier/
│   │   ├── classification.py
│   │   └── config.py
│   ├── object_detector/
│   │   └── detection.py
│   └── tracker/
│       ├── bytetrack.yaml
│       └── evaluate_f4k.py
└── README.md
```

* **`data/`**: Holds all the datasets used for training and evaluation.
* **`models/`**: Contains the final, trained model files (`.pt` and `.pth`).
* **`outputs/`**: The default location where all generated videos and data files are saved. This folder should be added to `.gitignore`.
* **`src/`**: Contains all the Python source code.

---

## 3. Model Preparation

The pipeline relies on two specialized models that must be trained beforehand:

1.  **Fish Tracker Model (`fish_detection.pt`)**: This is a single-class YOLOv8 model trained only to detect "fish". It provides stable tracking and should be trained on a dataset where all species labels have been converted to a single class (ID `0`).
2.  **Species Classifier Model (`fish_classification.pth`)**: This is a custom CNN model trained to identify specific fish species. It should is trained using the `src/classifier/classification.py` script.


---

## 4. Running the Pipeline

The main entry point for all operations is `src/main.py`. You must run all commands from the project's root directory. The pipeline can process both **pre-recorded video files** and **live camera footage**.

### Input Sources

The `--source` argument determines the input type:
* **For a video file**, provide the full path to the file (e.g., `"data/f4k/gt_113.mp4"`).
* **For a live camera**, provide its numerical ID (e.g., `"0"` for the default system webcam).

### Operational Modes

The pipeline has two modes, selected with the `--mode` flag:

* **`realtime`**: (Default) Optimized for speed. It classifies each fish the first time it's seen and displays the results immediately. Ideal for live camera feeds.
* **`buffered`**: A real-time capable mode that introduces a short delay. It only displays a track after it has been stable for a certain number of frames (`--min-duration`), reducing visual noise from fleeting detections.


### Example Commands

**Live Camera (Buffered Mode):**
```bash
# Use camera 0 and wait for 10 frames of stability
py src/main.py --source 0 --mode buffered --min-duration 10
```

> [!NOTE]  
> This project was developed using Python 3.10


---

## 5. Command-Line Arguments

You can customize the pipeline's behavior using the following arguments:

| Argument | Description | Default Value |
| :--- | :--- | :--- |
| **`--source`** | **[Required]** Path to the input video file or the camera ID (e.g., "0"). | `None` |
| `--yolo-model` | Path to the single-class 'fish' YOLO tracker model. | `models/fish_tracker.pt` |
| `--cnn-model` | Path to your trained CNN species classifier checkpoint. | `models/fish_classification.pth` |
| `--output-dir` | Directory where the annotated output videos will be saved. | `outputs` |
| `--mode` | The processing mode to use. | `realtime` |
| `--tracker-config`| Path to the tracker's `.yaml` configuration file for tuning. | `bytetrack.yaml` |
| `--min-duration` | Minimum frames a track must exist to be considered stable. (Used in `buffered` and `spatiotemporal` modes).| `5` |


