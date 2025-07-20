# Fish Tracking and Performance Evaluation Guide

## 1. What is Object Tracking?

While **Object Detection** finds objects in a single image (e.g., "there is a fish in this frame"), **Multi-Object Tracking (MOT)** goes a step further. It analyzes a sequence of frames to follow individual objects over time. The primary goal of tracking is to assign a **persistent ID** to each unique object. 

This project uses a powerful algorithm to achieve this, even in challenging underwater conditions.

---

## 2. The Tracking Algorithm: ByteTrack

The YOLOv8 framework, used in this project, integrates a state-of-the-art tracker called **ByteTrack**. It is highly effective because of its clever, two-stage approach to associating detections:

1.  **First Association (High Confidence)**: The tracker first looks at high-confidence detections from the YOLO model. It uses motion prediction (a Kalman filter) to estimate where a tracked fish should be in the new frame and matches it with these reliable detections.

2.  **Second Association (Low Confidence)**: This is ByteTrack's key innovation. Any tracks that were not matched in the first stage (e.g., a fish that is now blurry or partially hidden) are not immediately discarded. The tracker then looks at the low-confidence detections that were initially ignored. It tries to match these low-confidence boxes to the lost tracks.

This strategy is extremely effective for underwater video, as it allows the tracker to handle **occlusions** (fish swimming behind rocks or each other) and moments of **low visibility** without losing the object's ID.


---

## 3. Why Use the F4K Dataset for Evaluation?

Evaluating a tracking system requires a specific type of ground truth that goes beyond simple object detection. While many datasets provide bounding boxes for individual frames, very few provide the **persistent track IDs** needed to measure a tracker's ability to follow an object over time.

The *F4K (Fish4Knowledge)* dataset is one of the few publicly available resources that includes this crucial information.

* **Ground Truth for Tracking**: The primary reason for using F4K is that its annotations include a unique `trackingId` for each individual fish across multiple frames. This is the only way to quantitatively measure key tracking metrics like **ID Switches (`IDs`)** and **ID F1 Score (`IDF1`)**.

* **Challenging Real-World Scenarios**: The videos in F4K were specifically chosen for their difficult conditions (e.g., murky water, low contrast, crowded scenes), making it an excellent benchmark to test the robustness of our system.

---


## 4. How to Evaluate Performance

The evaluation is a two-step process. First, you run your pipeline on the F4K videos to generate results. Second, you run the dedicated evaluation script to compare those results against the ground truth.

### Step 1: Generate Tracker Results
Run your main pipeline (`src/main.py`) on all the F4K videos you want to test. This will create a `.txt` file for each video in your `outputs/logs/` directory.

### Step 2: Run the Batch Evaluation Script
Use the `evaluate_f4k.py` script to process all the results at once. It requires the path to the ground truth XML folder and the path to your tracker's output folder.

**Example Command (run from project root):**
```bash
py src/tracker/evaluate_f4k.py --gt-dir "Datasets/f4k/gt_bounding_boxes" --ts-dir "outputs/logs"
```

The script will automatically find all matching ground truth and result files, process them, and generate a final, overall performance summary.

---

## 5. Interpreting the Results

The evaluation script will print a summary table. Here is how to interpret the most important metrics:

| Metric | What it Means | Good Score | What a Bad Score Indicates |
| :--- | :--- | :--- | :--- |
| **`MOTA`** | **Overall Tracking Accuracy**. A composite score that combines false positives, misses, and ID switches. | **Higher is better.** | Your system is making many mistakes (missing fish, creating false tracks, or losing IDs). |
| **`IDF1`** | **ID F1 Score**. The best metric for measuring how well the tracker keeps the correct ID on a fish throughout its entire trajectory. | **Higher is better.** | Your tracker is frequently confusing different fish or losing and re-assigning IDs. |
| **`Rcll`** | **Recall**. "Of all the real fish in the video, what percentage did my detector find?" | **Higher is better.** | **This is often the main problem.** A low recall means your YOLO detector is failing to find the fish in the first place. The tracker cannot track what the detector doesn't see. |
| **`Prcn`** | **Precision**. "Of all the objects my system reported as fish, what percentage were actually fish?" | **Higher is better.** | Your detector is creating too many false positives (e.g., detecting rocks or seaweed as fish). |
| **`IDs`** | **ID Switches**. The total number of times a tracked fish's ID was incorrectly changed to a different one. | **Lower is better.** | The tracker's association logic is failing, likely due to occlusions or flickering detections. |
| **`FN`** | **False Negatives**. The total count of fish that were present in the ground truth but were missed by your system. | **Lower is better.** | Directly related to low Recall. Your detector needs to be improved. |
| **`FP`** | **False Positives**. The total count of incorrect detections. | **Lower is better.** | Directly related to low Precision. Your detector is too sensitive or confused by the background. |