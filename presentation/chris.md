
### Datasets

DeepFish is a benchmark that capture high variability of underwater fish habitats. Videos for DeepFish were collected for 20 habitats from remote coastal marine environments of tropical Australia. These videos were acquired using cameras mounted on metal frames, deployed over the side of a vessel to acquire video footage underwater. The video clips were captured in full HD resolution ( pixels) from a digital camera. In total, the number of video frames taken is 39,766 and their distribution across the habitats are shown in the image. 
The original labels of the dataset are only suitable for the task of classification. These labels were acquired for each video frame, and they indicate whether an image has fish or not (regardless of the count of fish). To address this limitation, we acquired point-level and semantic segmentation labels to enable models to learn to perform the computer vision tasks such as object counting, localization and segmentation. Point-level annotations are provided as a single click on each fish, and segmentation labels as boundaries over the fish instances. These annotations represent the (x, y) coordinates of each fish within the images and they are placed around the centroid of the corresponding fish. These annotations were acquired using Labelme, which is an open-source image annotation tool. We collected per-pixel labels for 620 images. We labeled the fish using layered polygons in order to distinguish between pixels that belong to fish and those to the background. The pixel labels represent the size and shape of the fishes in the image. We used Lear to extract these segmentation masks, an open-source image annotation tool commonly used for obtaining segmentation labels. We define a sub-dataset for each computer vision task: FishClf for classification, FishLoc for counting and localization, and FishSeg for the segmentation task. For each sub-dataset, we split the annotated images into training, validation, and test sets. Instead of splitting the data completely at random, we consider each split to represent the variability of different fish habitats and to have similar fish population size. Concretely, we first divide each habitat into images with no fish (background) and images with at least one fish (foreground). We randomly select 50% training, 20% for validation and 30% for testing for each habitat while ensuring that the number of background and foreground images are equal between them. As a result, we get a unique split consisting of 19,883, 7,953, 11,930 (training, validation and test) for FishClf, 1,600, 640, 960 for FishLoc, and 310, 124, 186 for FishSeg.

.
├── 7117
│   ├── train
│   └── valid
├── 7268
├── 7393
├── 7398
├── 7426
├── 7434
├── 7463
├── 7482
├── 7490
├── 7585
├── 7623
├── 9852
├── 9862
├── 9866
├── 9870
├── 9892
├── 9894
├── 9898
├── 9907
├── 9908
├── classes.txt
├── Nagative_samples
└── test


Text files (e.g., 9908_Epinephelus_f000165.txt)

- YOLO format annotations
- Each line represents one object: class_id x_center y_center width height
- All coordinates are normalized (0-1 range)

classes.txt

- Lists all fish species/classes in the dataset
- Maps class indices in the annotation files to species names

Negative_samples/

- Images containing no fish
- Important for training the model to avoid false positives
- Helps the detector learn what underwater scenes look like without fish

test/

- Held-out test set for model evaluation
- Ensures unbiased performance assessment

How tasks are performed with DeepFish:

1. Detection: YOLO directly uses the normalized bounding box coordinates from the .txt files
2. Classification: Built into the detection task - each bounding box has an associated class ID
3. Segmentation: Not supported (no mask annotations)
4. Tracking: Not directly supported (individual images rather than video sequences)


The F4K data is acquired from a live video dataset resulting in 27370 verified fish images. This data is organized into 23 groups, where the fish images and their masks are stored separately. Each cluster has a single package. The image files are named as "tracking id_fish id". Fish images with the same "tracking id" means they are belong to the same trajectory. "fish id" is a global unique id, which ranges from 1 to 27370. The representative image indicates the distinction between clusters shown in the figure below, e.g. the presence or absence of components (anal-fin, nasal, infraorbitals), specific number (six dorsal-fin spines, two spiny dorsal-fins), particular shape (second dorsal-fin spine long), etc. This figure shows the representative fish species name and the numbers of detections. The data is very imbalanced where the most frequent species is about 1000 times more than the least one. The fish detection and tracking software described in [1] is used to obtain the fish images. The fish species are manually labeled by following instructions from marine biologists [2].

f4k
 ┣ fish_image
 ┃ ┣ fish_01
 ┃ ┃ ┣ fish_000000009598_05281.png
 ┃ ┃ ┣ fish_000000009598_05283.png
 ┃ ┃ ┣ fish_000000009598_05285.png
 ┃ ┃ ┗ ...
 ┣ gt_bounding_boxes
 ┃ ┣ gt_106.xml
 ┃ ┣ gt_107.xml
 ┃ ┣ gt_109.xml
 ┃ ┣ gt_110.xml
 ┃ ┣ gt_111.xml
 ┃ ┣ gt_112.xml
 ┃ ┣ gt_113.xml
 ┃ ┣ gt_114.xml
 ┃ ┣ gt_116.xml
 ┃ ┣ gt_117.xml
 ┃ ┣ gt_118.xml
 ┃ ┣ gt_119.xml
 ┃ ┣ gt_120.xml
 ┃ ┣ gt_121.xml
 ┃ ┣ gt_122.xml
 ┃ ┣ gt_123.xml
 ┃ ┗ gt_124.xml
 ┣ mask_image
 ┃ ┣ mask_01
 ┃ ┃ ┣ mask_000000009598_05281.png
 ┃ ┃ ┣ mask_000000009598_05283.png
 ┃ ┃ ┣ mask_000000009598_05285.png
 ┃ ┃ ┗ ...
 ┗ videos
 ┃ ┣ gt_106.mp4
 ┃ ┣ gt_107.mp4
 ┃ ┣ gt_109.mp4
 ┃ ┣ gt_110.mp4
 ┃ ┣ gt_111.mp4
 ┃ ┣ gt_112.mp4
 ┃ ┣ gt_113.mp4
 ┃ ┣ gt_114.mp4
 ┃ ┣ gt_116.mp4
 ┃ ┣ gt_117.mp4
 ┃ ┣ gt_118.mp4
 ┃ ┣ gt_119.mp4
 ┃ ┣ gt_120.mp4
 ┃ ┣ gt_121.mp4
 ┃ ┣ gt_122.mp4
 ┃ ┣ gt_123.mp4
 ┃ ┗ gt_124.mp4

Fish_image/

- Contains the actual fish images extracted from underwater videos
- These are the input images for object detection
- Organized by fish species (fish_01, fish_02, etc.)
- File naming convention includes timestamp information (e.g., fish_000000009598_05281.png)

gt_bounding_boxes/

- XML files containing ground truth bounding box annotations
- Each XML corresponds to a video (gt_106.xml matches gt_106.mp4)
- These define the rectangular regions around fish in each frame
- Used for training and evaluating object detection models like YOLO

mask_image/

- Binary segmentation masks for each fish image
- White pixels represent the fish, black pixels represent background
- These provide pixel-level annotations for instance segmentation
- More precise than bounding boxes, showing the exact fish shape

videos/

- Original video files from which frames were extracted
- Enables temporal analysis and tracking across frames
- Can be used to generate additional training data or validate tracking algorithms

How tasks are performed with F4K:

1. Detection: YOLO uses the fish images with bounding box annotations from the XML files to learn to locate fish in images
2. Segmentation: The mask images provide pixel-level ground truth for training segmentation models
3. Classification: The folder structure (fish_01, fish_02) likely indicates different species classes
4. Tracking: The video files and frame naming convention allow tracking individual fish across consecutive frames

Final structure after preprocessing:
.
├── data.yaml
├── test
│   ├── images
│   ├── labels
│   └── labels.cache
├── train
│   ├── images
│   ├── labels
│   └── labels.cache
└── val
    ├── images
    ├── labels
    └── labels.cache

### YOLOv8 models

- **YOLOv8n**: This model is the most lightweight and rapid in the YOLOv8 series, designed for environments with limited computational resources. YOLOv8n achieves its compact size, approximately 2 MB in INT8 format and around 3.8 MB in FP32 format, by leveraging optimized convolutional layers and a reduced number of parameters. This makes it ideal for edge deployments, IoT devices, and mobile applications, where power efficiency and speed are critical. The integration with ONNX Runtime and TensorRT further enhances its deployment flexibility across various platforms

- **YOLOv8s**: Serving as the baseline model of the YOLOv8 series, YOLOv8s contains approximately 9 million parameters. This model strikes a balance between speed and accuracy, making it suitable for inference tasks on both CPUs and GPUs. It introduces enhanced spatial pyramid pooling and an improved path aggregation network (PANet), resulting in better feature fusion and higher detection accuracy, especially for small objects

- **YOLOv8m**: With around 25 million parameters, YOLOv8m is positioned as a mid-tier model, providing an optimal trade-off between computational efficiency and precision. It is equipped with a more extensive network architecture, including a deeper backbone and neck, which allows it to excel in a broader range of object detection tasks across various datasets. This model is particularly well-suited for real-time applications where accuracy is paramount, but computational resources are still a concern

- **YOLOv8l**: YOLOv8l boasts approximately 55 million parameters, designed for applications that demand higher precision. It employs a more complex feature extraction process with additional layers and a refined attention mechanism, improving the detection of smaller and more intricate objects in high-resolution images. This model is ideal for scenarios requiring meticulous object detection, such as in medical imaging or autonomous driving

- **YOLOv8x**: The largest and most powerful model in the YOLOv8 family, YOLOv8x, contains around 90 million parameters. It achieves the highest mAP (mean Average Precision) among its counterparts, making it the go-to choice for applications where accuracy cannot be compromised, such as in surveillance systems or detailed industrial inspections. However, this performance comes with increased computational demands, necessitating the use of high-end GPUs for real-time inference

Metrics:

- **mAPval**: mAPval stands for Mean Average Precision on the validation set.
It is a popular metric used to evaluate the accuracy of an object detection
model. Average precision (AP) is calculated for each class in the validation
set, and then the mean is taken across all classes to obtain the mAPval
score. A higher mAPval score indicates better accuracy of the object
detection model in detecting objects of different classes in the validation
set.

- **Speed CPU ONNX (ms)**: This relates to the object identification model's speed
while running on a CPU (Central Processing Unit) using the ONNX (Open
Neural Network Exchange) runtime. ONNX is a prominent deep learning
model representation format, and model speed can be quantified in terms of
inference time or frames per second (FPS). Higher values for Speed CPU
ONNX indicate faster inference times on a CPU, which can be important
for real-time or near-real-time applications.

- **Speed A100 TensorRT (ms)**: This refers to the speed of the object detection
model when running on an A100 GPU (Graphics Processing Unit) using
TensorRT, which is an optimization library developed by NVIDIA for deep
learning inference. Similar to Speed CPU ONNX, the speed can be
measured in terms of inference time or frames per second (FPS). Higher
values for Speed A100 TensorRT indicate faster inference times on a
powerful GPU, which can be beneficial for applications that require high
throughput or real-time processing.

- **FLOPs (B)**: FLOPs (B) stands for Floating point operations per second in
billions. It is a measure of the computational complexity of the model,
indicating the number of floating-point operations the model performs per
second during inference. Lower FLOPs (B) values indicate less
computational complexity and can be desirable for resource-constrained
environments, while higher values indicate more computational complexity
and may require more powerful hardware for efficient inference.

### AdamW

AdamW is a smarter version of Adam as it decouples weight decay from the gradient update step. Instead of adding weight decay to the loss function, it applies weight decay directly during the parameter update, leading to more consistent regularization and better generalization.

### Loss Function

- CIoU loss for bounding box regression to improve localization accuracy, represented as box_loss during training.
- DFL loss (Distribution Focal Loss), which you've rightly identified and is directly reported as dfl_loss. It helps the model to better estimate object categories.
- VFL loss (Varifocal Loss), which is not separately shown but is incorporated within cls_loss (class loss) in the training logs. VFL is designed to address imbalances and uncertainties in classification tasks.

Significance of Weights in Loss Functions: The weights (box, cls, dfl) dictate the emphasis the model puts on each component during training. For instance, box=7.5 puts substantial focus on getting the bounding box coordinates correct, while cls=0.5 and dfl=1.5 adjust the importance of class prediction accuracy and distribution of focal loss, respectively.

The default weights for the loss functions (7.5 for box, 0.5 for cls, and 1.5 for dfl) were determined through extensive experimentation and tuning to balance the contributions of each component to the overall loss effectively. You can indeed adjust these weights:

- Manipulating Loss Weights: If you're dealing with unbalanced classes, increasing the cls (class loss) might help. For example, trying values like cls=1 or cls=2 could proportionally increase the penalty for misclassifications, which may help correct class imbalance issues.

- Effects of Scaling Loss Weights: Increasing all losses by a common factor (like multiplying by 5 or 10) will not change the learning focus but might affect the convergence rate due to overall scale adjustment of gradients during backpropagation. It's usually more effective to adjust them relative to one another rather than scaling up all equally.

- Trade-offs: Indeed, there are trade-offs! Increasing one versus the others might make the model focus more on that aspect (e.g., more on getting bounding boxes right than classifying). Balancing this can be crucial depending on what’s more critical for your specific application

The CIoU (Complete Intersection over Union) loss indeed typically ranges between 0 and 1 for individual bounding boxes. However, the box_loss value you see during training is an aggregate measure, often a sum or mean over all bounding boxes in a batch. 

- CIoU Loss: Measures the overlap between predicted and ground truth boxes, considering aspect ratio and distance between box centers.
- Box Loss: In training logs, this is usually the sum of CIoU losses over all bounding boxes in a batch, hence it can exceed 1.

- Cls Loss (Varifocal Loss): This loss measures the classification error and can vary widely depending on the confidence scores and the number of classes. It doesn't have a strict range but is generally between 0 and 1 for individual detections.
- Dfl Loss (Distribution Focal Loss): This loss focuses on the localization precision of bounding boxes. Like the CIoU loss, its value for a single detection is typically between 0 and 1.

### NMS

