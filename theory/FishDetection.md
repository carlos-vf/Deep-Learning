# Fish detection task in literature

## Introduction
The task of automatic fish detection can be very useful both for enviromental and economic purposes. Manual routine monitoring of fishes across multiple habitats has always been done, for example through the capture of samples (destructive method) or through underwater visual census by divers (time consuming method).

To dimish the cost and time of this necessary task, many approaches to automatic fish detection have been tried in Machine Learning (ML) literature.

In general, automatic fish sampling involves the following three major tasks:
- Fish detection, which discriminates fish from non-fish objects in underwater videos. Non-fish objects include coral reefs, aquatic plants, sea grass beds, sessile invertebrates such as sponges, gorgonians, ascidians, and general background.

- Fish species classification, which identifies the species of each detected fish from the predetermined pool of different species.

- Fish biomass measurement, using length to biomass regression methods.

Many attempts were made to solve the first task with shallow learning architectures such as feature extraction through mathematical modelling. (Example: principal component analysis to extract key features)
A classical pipeline would have been composed of pre-processing on images, feature extraction (shape, texture and colour) and finally classification. 

However, these methods fail to correctly detect fishes in realistic and complex backgrounds. The luminosity of the image, the variation in the sea bed and the clarity of water are all factors that severely affect the accuracy of these models in real scenarios.

Thus the need and use of Artificial Neural Networks (ANN) for fish detection.

## Deep learning approaches to fish detection
- CNN + hierarchical feature
- CNN pre-trained on ImageNet + transfer learning (substituting the final layers and learning the parameters for those)
- Gaussian Mixture Models (GMM) to pre-process background/foreground + estimation of the optical flow on the input image + region-based CNN (Region Proposal Network)
- Fast R-CNN
- Faster R-CNN
It is important to notice that Fast R-CNN and Faster R-CNN are two-stage object detection algorithm: they first predict candidate boxes and then process each identified region to classify and refine the object boundaries.

## YOLO-Fish (2022)

As the dataset they joined together two datasets (DeepFish and OzFish). 
DeepFish didn't have any bounding box annotation so they added correct labelling using labelImg tool. Many frames have multiple fishes at a time.

To construct the training and test set they made a random stratified selection (where they take into account the different habitats). 

The YOLO-Fish model is a modified version of the YOLOv3.
The YOLO model already existed to make general object detection. It is a region based convolutional neural network that combines region proposal network branch and classification stage into a single network. It is a "proposal-free" method, thus directly predicting bounding boxes. YOLOv2 used the anchors from Faster-RCNN. YOLOv3 has Darknet-53 as backbone network, an upsampling network and three detection heads similar to the idea of Feature-Pyramid Network (FPN).

The input is an image of 608x608 pixels, it is passed through the Darknet-53 network, then through a Spatial Pyramid Pooling (SPP) and then finally they obtain three different scale models (small, medium and large).

With respect to the basic YOLOv3 model the YOLO-Fish is able to detect tiny objects in the dataset. They did not modify subsequent versions for their increased model complexity.

They note that it might be interesting to apply attention based models/transformers.

## Subsequent models

The first authors to apply the YOLO model to fish detection were actually Sung et al. in 2017, where they just applied YOLO without modifying anything. 

In 2024 the YOLOv5 model was modified to make marine recognition (specifically recognition of marine zoobenthos). The authors added a lightweight feature extraction network as backbone, a bottleneck transformer and a Convolutional Block Attention module (CBA). They also applied an underwater image enhancement algorithm based on color balance and multi-input fusion.

In 2023 the YOLOv5s (variant of YOLOv5) was modified by adopting transformer self-attention and coordinate attention.

In 2025 the YOLOv10 was combined with DeepSORT algorithm to count and detect fishes in video.

There have also been other experiments that mix Transformers and CNN: transformers are generally more effective, but they are also slower which is the reason they are often mixed together.

Some experiments have also been done on specific fish type recognition: in 2025 a YOLOv5 model was trained based on fish jumping behaviour, achieving high performance thanks to feature extraction (like the size and colour of the jumping fishes). By using this technology on a more limited dataset they were able to obtain very high precision, highlighting the importance of a good dataset.


## Datasets

Datasets used in literature:
- LifeCLEF 2014/2015 (10/15 species)
- ImageCLEF
- main resource: Fish4Knowledge (videos)
- Rockfish (problematic)
- QUT (problematic)
- DeepFish
- OzFish

## Model performance evaluation
For the task of species classification we can use Precision, Recall or the Mean Average Precision (MAP).

We can use these metrics also for fish detection, by considering as a False Positive a box is labeled as fish when it was not a fish and a False Negative when a fish box was not identified.  

## References:
- Salman et al. "Fish species classification in unconstrained underwater environments based on deep learning" (2016)
- Siddiqui et al. "Automatic fish species classification in underwater videos: exploiting pre-trained deep neural network models to compensate for limited labelled data" (2018)
- Salman et al. "Automatic fish detection in underwater videos by a deep neural network-based hybrid motion learning system" (2020)
- Li and Cao "Fast Accurate Fish Detection and Recognition of Underwater Images with Fast R-CNN" (2015)

- "YOLO-Fish: A robust fish detection model to detect fish in realistic underwater environment" (2022)

- Liu et al. "Two-Stage Underwater Object Detection Network Using Swin Transformer" (2022)
- Liu et al. "Underwater Object Detection Using TC-YOLO with Attention Mechanisms" (2023)
- Zhang et al. "Marine zoobenthos recognition algorithm based on improved lightweight YOLOv5" (2024)
- Khiem et al. "A novel approach combining YOLO and DeepSORT for detecting and counting live fish in natural environments through video" (2025)
- Li et al. "Enhancing aquatic ecosystem monitoring through fish jumping behavior analysis and YOLOV5: Applications in freshwater fish identification" (2025)

Articles for the YOLO architecture and its modules:
- Redmon et al. "You only look once: unified, real-time object detection." (2016)
- Redmon and Farhadi "Yolo9000: better, faster, stronger." (2017)
- Ren et al. "Faster r-cnn: towards realtime object detection with region proposal networks." (2015)
- Joseph and Ali "Yolov3: an incremental improvement." (2018)
- Lin et al. "Feature pyramid networks for object detection." (2017)
- Bochkovskiy et al. "Yolov4: optimal speed and accuracy of object detection." (2020)
- He et al. "Spatial pyramid pooling in deep convolutional networks for visual recognition." (2015)
