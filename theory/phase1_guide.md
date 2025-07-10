# Phase 1: Fish Density Estimation - Minimal Resources Implementation

## Overview
Build a proof-of-concept fish tracking and counting system using minimal computational resources, focusing on demonstrating the core concepts before scaling up.

## Hardware Requirements (Minimal)
- **CPU**: Any modern quad-core processor (Intel i5/AMD Ryzen 5 or equivalent)
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: Optional (can run entirely on CPU for testing)
- **Storage**: 20GB free space
- **Camera**: Webcam or smartphone camera for testing

## Software Stack

### Core Dependencies
```bash
# Create virtual environment
python -m venv fish_tracking_env
source fish_tracking_env/bin/activate  # Linux/Mac
# fish_tracking_env\Scripts\activate  # Windows

# Install lightweight packages
pip install ultralytics==8.0.196  # YOLOv8
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
pip install pandas==1.5.3
pip install matplotlib==3.7.2
pip install pillow==10.0.0
pip install sort-track  # Simple SORT implementation
```

## Phase 1 Implementation Strategy

### Step 1: Setup and Data Collection (Week 1)
**Goal**: Get basic detection working with minimal data

1. **Download a pre-trained YOLOv8n model** (3MB, CPU-friendly)
2. **Create a small test dataset**:
   - Record 2-3 short videos (30-60 seconds each) of fish
   - Use aquarium videos from YouTube if no access to real fish
   - Extract 200-300 frames manually
3. **Manual annotation** using free tools:
   - [LabelImg](https://github.com/tzutalin/labelImg) (free, easy to use)
   - Label only 100-150 images initially
4. **Test basic detection** on unlabeled frames

### Step 2: Simple Tracking Implementation (Week 2)
**Goal**: Add basic tracking without training custom models

1. **Integrate SORT tracker** with pre-trained YOLOv8n
2. **Implement basic counting logic**:
   - Count unique track IDs over time windows
   - Simple line-crossing detection
3. **Test on recorded videos**
4. **Measure basic performance** (detection rate, tracking stability)

### Step 3: Basic Density Estimation (Week 3)
**Goal**: Convert counts to density metrics

1. **Manual calibration**:
   - Measure physical area in your test videos
   - Calculate pixels-to-real-world conversion
2. **Implement density calculation**:
   - Fish count per frame ÷ area = instantaneous density
   - Moving average over time windows
3. **Basic visualization**:
   - Simple plots of density over time
   - Annotated video output

### Step 4: Evaluation and Documentation (Week 4)
**Goal**: Measure what works and what doesn't

1. **Manual ground truth counting** on test videos
2. **Calculate accuracy metrics**:
   - Mean Absolute Error (MAE)
   - Percentage error vs manual counts
3. **Document limitations and challenges**
4. **Plan Phase 2 improvements**

## Key Implementation Details

### Minimal YOLOv8 Setup
```python
from ultralytics import YOLO
import cv2

# Load pre-trained nano model (CPU-friendly)
model = YOLO('yolov8n.pt')  # Automatically downloads 6MB model

# Configure for CPU inference
model.to('cpu')
results = model(source='path/to/video.mp4', 
               conf=0.3,  # Lower confidence for more detections
               classes=[14],  # COCO class for 'bird' (closest to fish)
               save=False,
               stream=True)
```

### Simple Tracking Integration
```python
from sort import Sort
import numpy as np

tracker = Sort(max_age=30, min_hits=3)
track_history = {}

for frame_idx, r in enumerate(results):
    # Convert YOLO detections to SORT format
    detections = []
    if r.boxes is not None:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            detections.append([x1, y1, x2, y2, conf])
    
    # Update tracker
    if len(detections) > 0:
        tracks = tracker.update(np.array(detections))
        # Count unique track IDs
        unique_fish = len(tracks)
```

## Expected Outcomes - Phase 1

### Performance Targets (Realistic for minimal setup)
- **Detection Accuracy**: 60-75% (compared to manual counting)
- **Processing Speed**: 5-15 FPS on CPU
- **Tracking Stability**: Track fish for 3-5 seconds continuously
- **Density Estimation**: ±20% error compared to manual counting

### Deliverables
1. **Working prototype** that processes videos end-to-end
2. **Small annotated dataset** (100-200 images)
3. **Basic performance evaluation** with quantified metrics
4. **Documentation** of what works and limitations
5. **Plan for Phase 2** with identified improvements

## Cost Breakdown

### Free Resources
- Pre-trained YOLOv8n model
- SORT tracking algorithm
- LabelImg annotation tool
- OpenCV and basic Python libraries
- YouTube aquarium videos for testing

### Minimal Costs (Optional)
- **Cloud compute for testing**: $5-10 (Google Colab Pro if needed)
- **Small dataset purchase**: $0-50 (if you buy labeled fish dataset)
- **Basic webcam**: $20-50 (if you don't have one)

## Common Challenges & Solutions

### Challenge 1: "My detection accuracy is low"
**Solutions**:
- Lower confidence threshold (0.2-0.3)
- Use different pre-trained classes (try 'bird', 'person', or retrain)
- Improve video quality (better lighting, stable camera)

### Challenge 2: "Tracking keeps losing fish"
**Solutions**:
- Tune SORT parameters (max_age, min_hits)
- Try simpler IoU-based tracking
- Process at lower frame rates (every 2nd frame)

### Challenge 3: "Running too slowly"
**Solutions**:
- Reduce video resolution (480p instead of 1080p)
- Process every nth frame instead of all frames
- Use YOLOv8n instead of larger models

### Challenge 4: "Hard to get test data"
**Solutions**:
- Use aquarium livestreams from YouTube
- Visit local aquarium for video recording
- Start with simpler scenarios (clear water, good lighting)

## Timeline (4 Weeks, Part-time)

**Week 1**: Environment setup + basic detection testing  
**Week 2**: Add tracking + test integration  
**Week 3**: Implement counting logic + basic density calculation  
**Week 4**: Evaluation + documentation + Phase 2 planning

## Success Criteria
- [ ] System processes video from start to finish without crashing
- [ ] Can detect fish in at least 60% of frames where fish are visible
- [ ] Tracks individual fish for at least 3 seconds
- [ ] Produces density estimates within 25% of manual counts
- [ ] Documented understanding of current limitations
- [ ] Clear plan for Phase 2 improvements

## Next Steps to Phase 2
Based on Phase 1 results, Phase 2 will focus on:
- Custom model training with your specific dataset
- GPU acceleration for real-time performance
- More sophisticated tracking algorithms
- Improved accuracy through fine-tuning
- Edge deployment optimization
