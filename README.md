# Phase 3: Real-Time Video Tracking & Analysis

## Overview
Phase 3 leverages the custom fish detection model built in Phase 2 to perform robust, real-time tracking and analysis on video files. The focus shifts from static image detection to understanding object behavior over time, extracting meaningful data, and producing actionable insights from your underwater videos.

---

## Phase 3 Goals
- **Robust multi-object tracking** with persistent IDs for individual fish.
- **Video analytics pipeline** that extracts fish counts and track data.
- **Maintain real-time performance** (>15 FPS) during video processing.
- **Generate insightful visualizations** from the extracted tracking data.

---

## Hardware Requirements (Phase 3)
Hardware requirements are for **inference**, which is less demanding than training.
- **GPU**: GTX 1060 / RTX 2060 or better (4GB+ VRAM recommended for smooth playback).
- **RAM**: 16GB minimum.
- **Storage**: 20GB free space for videos and output data.

---

## Software Stack Upgrade
This phase introduces libraries for video processing, tracking, and data analysis.

### Core Dependencies
```bash
# Core library remains the same
pip install --upgrade ultralytics

# Add libraries for video and data handling
pip install --upgrade opencv-python
pip install --upgrade pandas numpy
pip install --upgrade scikit-learn # For metrics
pip install --upgrade seaborn matplotlib # For plotting

# Verification
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

### Week 1: Baseline Video Inference & Tracking
1.  **Develop a script** to run the Phase 2 model on a single video file.
2.  **Integrate a default object tracker** (e.g., ByteTrack, which is built into YOLOv8).
3.  **Process a test video** and save the output with bounding boxes and track IDs.
4.  **Establish baseline metrics**: Frames Per Second (FPS) and initial tracking quality.

### Week 2: Tracking Optimization & Data Extraction
1.  **Tune tracker parameters** (e.g., confidence thresholds) to reduce ID switches.
2.  **Implement logic to handle occlusions** and re-identification.
3.  **Extract raw tracking data**: Frame number, track ID, bounding box coordinates.
4.  **Save the extracted data** to a structured format (e.g., a CSV file).

### Week 3: Data Analysis & Visualization
1.  **Develop scripts to analyze** the generated tracking data from the CSV files.
2.  **Calculate key video-level metrics**:
    * Total unique fish count.
    * Duration each fish is on-screen.
    * Fish count per frame.
3.  **Create visualizations**:
    * A plot of fish count over time for an entire video.
    * A histogram of track durations.

### Week 4: Final Evaluation & Packaging
1.  **Run the complete pipeline** on all test videos.
2.  **Compare final fish counts** against your ground truth annotations.
3.  **Benchmark final FPS** and document tracking stability.
4.  **Refactor the code** into a clean, reusable application/script.


## Project Structure for Phase 3

```
deepfish_project/
├── input_videos/
│   ├── test_video_01.mp4
│   └── ...
├── output_data/
│   ├── test_video_01_tracks.csv  # CSV with tracking data
│   └── ...
├── output_videos/
│   ├── test_video_01_tracked.mp4 # Video with boxes and IDs
│   └── ...
├── models/
│   └── best.pt                   # Your trained model from Phase 2
├── src/
│   ├── track_video.py            # Main script for this phase
│   └── analyze_tracks.py         # Script for data analysis
└── README.md
```

## Key Phase 3 Features

### 1. **Multi-Object Tracking (MOT)**
- Uses a tracker like **ByteTrack** to associate detections from consecutive frames.
- Assigns a unique and persistent **Track ID** to each fish, allowing you to follow it through the video.

### 2. **Video Data Extraction**
- Moves beyond just drawing boxes on a video.
- Creates a structured log (CSV) of every detection, linked to a specific fish (via its track ID) and a specific moment in time (the frame number).

### 3. **Performance Optimization for Video**
- Focuses on maintaining a high FPS rate by optimizing the video processing pipeline.
- Techniques include efficient video I/O with OpenCV and running inference in a streamlined loop.

### 4. **Automated Video Analytics**
- Builds a repeatable process to turn raw video into structured data and visualizations.
- Allows for consistent analysis across all of your video assets.


## Evaluation Metrics
Metrics shift from static image accuracy (mAP) to tracking and video-level accuracy.

- **Fish Count Accuracy**: Final count vs. ground truth (e.g., `±10% error`).
- **Track Stability**: Number of ID switches per fish (lower is better).
- **Processing Speed**: **Frames Per Second (FPS)** on your target hardware.
- **Advanced (Nice to Have)**: Multiple Object Tracking Accuracy (MOTA) and IDF1 scores, which require detailed per-frame tracking annotations.


## Expected Phase 3 Outcomes

### Performance Targets
- **Tracking Speed**: 15-40 FPS on GPU.
- **Fish Count Accuracy**: Within ±10% of ground truth on test videos.
- **ID Switches**: Minimized for a majority of tracks.

### Deliverables
1.  **An optimized video processing script** (`track_video.py`).
2.  **Tracked output videos** showing bounding boxes and persistent track IDs.
3.  **CSV files** containing detailed, frame-by-frame tracking data.
4.  **An analysis script** (`analyze_tracks.py`) that generates summary statistics and plots.
5.  **An evaluation report** summarizing the tracking performance across all videos.


## Phase 3 vs. Phase 2 Comparison
| Metric                | Phase 2 (Training)                      | Phase 3 (Tracking & Analysis)         |
| --------------------- | --------------------------------------- | ------------------------------------- |
| **Primary Goal** | Create an accurate image detector       | Analyze object behavior in video      |
| **Input** | Annotated images (frames)               | Trained model (`.pt`) and video files |
| **Output** | A trained model (`.pt`)                 | Tracked videos and data (CSV)         |
| **Key Metric** | mAP (Mean Average Precision)            | FPS, Fish Count Accuracy, MOTA/IDF1   |
| **Core Technology** | Transfer Learning                       | Multi-Object Tracking (e.g., ByteTrack)|


## Common Phase 3 Challenges & Solutions

### Challenge 1: "Lost Tracks / Frequent ID Switching"
**Solutions**:
- **Tune tracker thresholds**: Adjust the confidence scores required to initialize or maintain a track.
- **Adjust detector confidence**: A lower detection confidence (`conf`) can help the tracker see objects in challenging frames, but may increase false positives.
- **Analyze failure cases**: Visually inspect videos where IDs switch to understand the cause (e.g., fast motion, occlusions, similar-looking fish).

### Challenge 2: "Inference is Too Slow for Real-Time"
**Solutions**:
- **Reduce processing resolution**: Process the video at a lower resolution (e.g., 640p instead of 1080p).
- **Frame skipping**: Process every Nth frame instead of every single frame. This is effective for slow-moving scenes.
- **Use a lighter model**: If not already using it, switch to the `yolov8n` model from Phase 2.
- **Model Optimization (Advanced)**: Export the `.pt` model to a faster format like ONNX or TensorRT.

### Challenge 3: "Tracker is Confused by False Detections"
**Solutions**:
- **Increase the detection confidence threshold** in your tracking script to filter out low-confidence detections from the model.
- **Retrain the Phase 2 model**: If certain non-fish objects (e.g., seaweed, reflections) are consistently detected, add these as negative examples to your training data and retrain the model.