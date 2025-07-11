# Summary: The Brackish Multi-Object Tracking (MOT) Dataset

[Kaggle](https://www.kaggle.com/datasets/aalborguniversity/brackish-dataset?)
[Paper](https://vap.aau.dk/the-brackish-dataset/)

The Brackish dataset is the first publicly available European underwater video dataset with bounding box annotations. It was recorded in a brackish strait in Denmark using a fixed camera setup mounted 9m below the surface.

Initially created for object detection, it has been significantly updated to become a benchmark for Multi-Object Tracking.

---

### Key Features

* **Content**: A total of **98 videos** (89 original + 9 new MOT sequences).
* **MOT Expansion**: A 2023 update added new sequences and ground truth annotations in the standard **MOTChallenge format**, which is perfect for evaluating trackers.
* **Annotated Classes**: There are 6 categories of marine life:
    * `fish`
    * `small_fish`
    * `crab`
    * `shrimp`
    * `jellyfish`
    * `starfish`
* **Multiple Annotation Formats**: The dataset provides annotations in several widely-used formats:
    * YOLO Darknet
    * MS COCO
    * AAU Bounding Box
    * MOTChallenge
* **Major Annotation Update**: The dataset was updated in August 2020 to fix false negatives, adding approximately 14,000 new annotations. It is recommended to use the latest version.

---

### Usage & Tools

* **Data Splits**: The data is pre-split into `train.txt`, `valid.txt`, and `test.txt` files, following an 80/10/10 ratio.
* **Provided Scripts**: The dataset comes with utility scripts to:
    * Extract frames from videos using `ffmpeg`.
    * Convert between different annotation formats.
* **Academic Reference**: The project is well-documented in two papers, one for the original dataset (CVPRW 2019) and one for the tracking expansion, "BrackishMOT: The Brackish Multi-Object Tracking Dataset" (SCIA 2023). If you use this dataset, it is best practice to cite their work.