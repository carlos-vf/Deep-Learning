# YOLO architecture and theory

[YOLO timeline](YOLO_timeline.pdf)

## Introduction
YOLO is a fast object detection architecture based on Convolutional Neural Networks that works extremely fast thanks to its *proposal-free* scheme (also called single-shot detection).

Instead of predicting a class (the presence of a ceratin object) given a subsection of the image, YOLO predicts the bounding box coordinates from the whole image. Object detection is thus reframed as a regression problem, which makes the prediction much faster, at the cost of slightly less accuracy.

Also, since the CNN is given the whole image during training and not just some pre-processed subparts of it, the network can encode contextual informations about the class to detect. This reduces the number of background errors.

Finally, YOLO is able to generalize better. Object detection works on artwork even if it was trained on real images.

All these reasons make YOLO a better candidate for object detection.

## Before YOLO:
The two main approaches were to either evaluate classifiers at various locations of an image, like Deformable Parts Models (DPM), or to generate potential bounding boxes and then evaluate a classifier on those boxes, *proposal-based* models like Recurrent Convolutional Neural Networks (R-CNN).

DPM use a sliding window approach, they have a disjointed pipeline where they first extract static features, classify regions and then predict bozes on regions. Because of all these passages they are much slower than YOLO.

R-CNN first generate potential bounding boxes, then a CNN extracts features, a SVM scores the boxes, a linear model adjusts the bounding boxes and a non-max suppression eliminates duplicate detections. Again, the large number of stages in the pipeline lead to a slow system to train. 

There had been proposed also Fast and Faster R-CNN, variants where the proposal regions are obtained through neural networks. Still, the division in phases makes the computation slower than what it is.

In general, a CNN is a network made of **convolutional layers**, each of which is equivariant to translation (which means that if I shift the input of the layer the output gets shifted in the same way). This is useful to exploit the property of the ouput being equivariant with respect to translation in image segmentation and other similar tasks.

It often includes also pooling mechanisms (methods for scaling down the representation size) to induce partial invariance to translation (which means that if I shift the input of the layer the pooling method may give the same output). This is useful to exploit the property of the output being invariant with respect to translation in image classification and other similar tasks.

The weights of the convolution operation (it is actually a cross-correlation instead of a convolution) form the **convolution kernel** or **filter**.
Each convolutional layer is characterized by its **kernel size** (number of inputs of which to compute the convolution), its **stride** (the frequency with which we compute the convolution)  and **dilation rate** (the frequency with which we take the inputs).

Example: a stride of 2 halve the subsequent layer, a large kernel size requires more weights and a large dilation takes in the convolution input nodes that are further from each other.

## YOLOv1 (2015)

First, the image is divided in an $S \times S$ grid. If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object.

Each grid cell predicts **B bounding boxes** and **confidence scores** for those boxes. These confidence scores reflect how confident the model is that the box contains an object and also how accurate it thinks the box is that it predicts. A bounding box is described by 4 numbers: the coordinates of the center, its width and height.

Ideally, we want the confidence score to be equal to the ratio of intersection area over union area (**IOU**) between the predicted box and the ground truth. It is easy to see that, if the two boxes overlaps exactly, the intersaction will be equal to the union (ratio = 1) while if the two boxes are completely separated the intersaction, and ratio, will be 0.

Each grid cell also predicts **C conditional class probability** (conditioned on the presence of an object in that grid). 

This means that in total the predictions are contained in a tensor of dimension:
$$S \times S \times (B\cdot 5 + C)$$
(for every grid there are $B$ boxes and $C$ probabilities, and each box has 4 numbers and an object confidence score)

The network was inspired by GoogLeNet model and it contains 24 convolutional layers followed by 2 fully connected layers.

(There is also a fast version that has just less layers)

The input is of a $448 \times 448$ RGB image (double the resolution of a standard $224 \times 224$ RGB image). It is divided in a $7 \times 7$ grid (so the image is divided in $49$ cells of $4096$ pixels each).

First, we have a convolutional layer where the kernel has size $7 \times 7 \times 64$ and stride $2$ and a maxpool layer of size $2$. Since the stride is $2$ the resulting layer has height and width halved, but also since the kernel has multiple channels the final layer has $64$ channels. The max pooling also halves the representation and another convolutional kernel raises the number of channels, obtaining the layer $112 \times 112 \times 192$.

A similar sequence of layers (convolutional layers that augment the channels and maxpool layers that halve the representation size) follows until the last 2 fully connected layers. The first FC brings the channels to 1 and the last obtains the $7 \times 7 \times 30$ tensor of predictions. 

To train the network, they first pretrain for classification the first 20 convolutional layers followed by a average-pooling layer and a fully connected layer using the ImageNet dataset. Then they convert the model by adding the 4 convolutional layers and the 2 fully connected layers and train the new network with the weights initialized by the pretraining.

All outputs are between 0 and 1 since we parametrize the coordinates and lenghts to be relative measures with respect to the image.

They use sum-squared error as the loss to minimize, but with lower weight to classification errors in cells that do not contain objects. 

Given a cell $i$ (of the total of $49$) and a bounding box $j$ (of the $2$ in the paper), we have a flag that tells us of that predictor in that cell is "responsible" for that prediction. (Of course, we can't expect one box to be valid for multiple predictions or the loss would always be high)

The final loss contains:
- a term that penalizes errors in the coordinate predictions
- a term that penalizes errors in the size of the bounding box
- a term that penalizes the object confidence score
- a term that penalizes the object confidence score when there are actually no objects
- a term that penalizes errors in the probability distribution over classes

Thanks to the "flag" variables we avoid penalizing cases that don't make sense (like classification error when there is no object).

They train for 135 epochs with batch size 64, momentum of 0.9 and decay of 0.0005. They start by slowly raising the learning rate and then lower again. They also use dropout (during training some units are clamped to 0) and data augmentation (dataset augmentated by scaling, translation and modifying exposure and saturation).

RMK: for large objects YOLO could predict multiple boxes, this can be solved by non-maximal suppresion.

RMK: just like R-CNN also YOLO by dividing the image in cells is "proposing" regions for the bounding boxes. However the proposal regions are far fewer and they are not processed in two stages but directly by the network.

### Weaknesses
- YOLO struggles in identyfing small objects. This is because it imposes strong spatial constrains: each cell has only two boxes and can only have one class. It can't detect multiple small fishes next to each other
- YOLO has multiple downsampling layers, which means that predictions are based on very blurry features
- To partially differentiate between errors in large boxes and in small boxes they also predict the square root of the size of the box instead of the size directly. However this is not enough, the main source of error is still incorrect localizations.

## YOLOv2 (2016)

The main idea is to improve recall and localization of YOLOv1 while maintaining classification accuracy.

The main change is considered to be the introduction of anchors. This is done to better handle objects at different scales and aspect ratios.

Since they want to keep the model fast they don't want to just scale up the network. Instead, they apply various ideas to the YOLOv1 network:
- **Batch normalization**: each activation is shifted and rescaled so that the mean and variance across the batch are the same and can be learned by the network, this method can help regularize the model. This way, they **remove the dropout method** (since its purpose was regularization too).
- **High resolution classifier**: instead of training a classifier on $224 \times 224$ images and then train an object detector on $448 \times 448$ images, in between they add a step where a new classifier is fine-tuned on $448 \times 448$ images. This way the network doesn't change at the same time the resolution and the task to perform.
- **Convolutional with anchor boxes**: instead of predicting coordinates of the boxes, they predict offset w.r.t. anchor boxes previously defined. To do so they eliminate one pooling layer (higher resolution) and change input so that the output is going to have $13$ cells. The output in this case is a feature map (equivalent to a channel) that contains the offsets. They want an odd number of output cells so that there is only one cell for the center of the image. Also the boxes in one grid cell can predict different classes (in YOLOv1 they all had to predict the same class).
- **Dimension clusters**: To apply the anchor boxes we would need to hand pick their dimension. To choose them they run k-means clustering on the bounding boxes of the training set. Each cluster is a proposal anchor box, so it represents the number of boxes that will be used as prior. 
- **Direct location prediction**: If they just applied anchor boxes and offset prediction then the originally predicted boxes can end up anywhere, leading to high instability in the predictions. Instead, they use anchor boxed but they make the prediction of the center of the boxes be relative to the grid cell dimension. This way the final boxes will not be too far from the prior ones (where the prior are the anchor boxes).
- **Fine-grained features**: they add a passthrough layer that is concatenated with a subsequent lower resolution layer. They are of the same dimension but they are concatenated as many channels.
- **Multi-scale training**: Every 10 batches the input images dimensions are changed. This make the network learn to generalize better to different images size.
- **Darknet-19 as classification network**: While YOLOv1 used a variant of GoogleNet they now propose a new base architecture. This network has 19 convolutional layers and 5 maxpooling layers. For every pooling step the number of channels is doubled, just like in VGG models (the most common architectures for feature extraction). It is faster than VGG while mantaining a good accuracy.

**Training process**:
They first train on imageNet for classification for 160 epochs, using data augmentation. For the fine tuning of classification on higher resolution images they only consider 10 epochs.
Finally, they remove the last convolutional layer and add 3 convolutional layers, each with 1024 filters (a filter is a kernel, each of them produce an output channel, thus the output layers of the convolution have 1024 channels), and a final $1 \times 1$ convolutional layer. In the paper they predict 5 boxes with 5 coordinate each and 20 classes so the last convolutional layer will have: $(5*(5+20))=125$ channels. Here it is added the passthrough layer. The output tensor will have dimension: $S\times S\times 125$. They train the network for 160 epochs.

**Joint training**: During the training they also mix images from both detection and classification datasets. When the network sees an image for detection the full loss is optimized. When it sees an image for classification the classification loss is optimized. This is doable because the detection datasets usually are detecting the same class and thus it can easily be added to the dataset. Also this helps with the main problem of dataset size difference between the two tasks.

The problem with mixing datasets is that some classes are **not mutually exclusive** (example: dog and husky can be applied to the same image, the network shouldn't have to learn to discriminate between the two).
Possible solution: **hierarchical classification**. They create a WordTree and at each node they learn a probability distribution over the subcategories. The root of the tree is the class "physical object".

In the end, the mixed training make the network be able to generalize also on new categories well, for example on new animals.

RMK: COCO dataset is made of mutually exclusive classes.

## YOLOv3 (2018)

They just add some cool and good ideas from other people work.

The main change is considered to be the feature pyramid network that allows the model to predict at 3 different scales.

- Predicts also an objectness score for each bounding box.
- Instead of using softmax they use **indipendent logistic classifiers**. This also solves the problem of not mutually exclusive datasets.
- They upgrade the feature extractor to **Darknet-53**. It contains 53 convolutional layers (hence the name) as well as residual layers ("shortcut connections") and pooling laters.
- They predict boxes at **3 different scales**. The scale determines the number of grid cells the image is divided into. After the base feature extractor (darknet) they add several convolutional layers, the last predicts a 3-d tensor that contains all predictions. By using 3 bounding boxes (each at a different scale) they obtain $N\times N \times (3 \cdot (4+1+80))$ (where N changes based on the scale), where there are $5$ numbers for each box, $80$ classes and $3$ boxes for each grid cell. They take the feature map and upsample it and concatenate with a previous one. They also add more convolutional layers. They used 9 cluster as prior boxes (3 for each scale). The predictions are made for the three scales.
- They use multi-scale training, data augmentation, batch normalization and "all the standard stuff".

**Results**: better performance, suddently better at detecting small objects. Still not very good on getting the exact box.

**Things that didn't work**:
- standard anchor box and offset prediction with linear activation
- linear predictions for the center of the boxes (better logistic activation)
- focal loss: since many times the proposal anchor boxes do not actually contain objects this lead to class imbalance (too many negatives and few positives). Focal loss adds an extra parameter in the loss that down-weights the effect of weel-classified examples to improve performance 
- dual IOU thresholds

## Parenthesis: Feature Pyramid Networks (FPN)

Since in YOLOv3 they start predicting at 3 different scales, we should at least spend a word on a similar concept: feature pyramid networks.

At first, objecton detection models tried to improve accuracy by using **featurized image pyramids**: they processed the same image at different scales, applying a feature extractor independently at each scale. This was very slow and expensive.

Instead, they start from the original image, learn to extract features through a CNN to get to the lowest resolution feature map. Then they create a new feature pyramid from up to bottom by upsampling from the low-resolution feature maps. This process is enhanced by some lateral connections to the high-resolution feature maps.

## Parenthesis: some general annotations on object detectors and ways to improve them

Object detectors are usually composed of two parts: a **backbone** (like DarkNet53) which is pre-trained on ImageNet (huge dataset popular in those years) and a **head** which is used to predict classes and bounding boxes of objects.
The head can be a "one-stage detector"/"dense prediction" (like YOLO) or a "two-stage detector"/"sparse prediction" (like R-CNN).

Object detectors developed in recent years usually insert some layers between backbone and head to collect feature maps from different stages. These layers form the **neck** of the architecture. Often they contain several bottom-up paths and top-down paths, like Feature Pyramid Networks.

**Bag of freebies (BOF)** are all the methods that chang the *training* cost/strategy without affecting the *inference* cost/strategy. Some common examples include:
- **Data augmentation** is used to improve the robustness of the model. Pixel-wise adjustments include photometric distortions (adjust the brightness, contrast, hue, saturation and noise) and geometric distortions (add random scaling, cropping, flipping and rotating). Other ways would be to simulate object occlusion issues, like random erase or adding a mask, or to mixup different images.
- **Focal loss** is used to deal with the problem of data imbalance between different classes.
- **Labeling smoothing** is used to express the relationship of the degree of association between different categories, instead of using the one-hot hard representation. (OFC, in our case it would mean to actually understand the relation between different fish species...)
- Modification of the **objective function of Bounding Box (BBox) regression** is done to avoid problems like the regression loss increasing with the scale or the predicted objects losing integrity because the coordinates are predicted independently. There have been proposed various losses based on modification of the IOU metric (like GIOU, DIOU or CIOU). They can lead to faster convergence and better accuracy.

**Bag of specials (BoS)** are all methods that only increase the inference cost by a small amount but can significantly improve the accuracy of object detection. They usually improve some specific attributes in a model. Some common examples include:
- Modules that **enhance receptive field** like **Spatial Pyramid Pooling (SPP)**.
- Modules that introduce an **attention mechanism** like **Squeeze-and-Excitation (SE)** or **Spatial Attention Module (SAM)**. SE will actually raise the inference too much, but SAM doesn't affect inference time at all, just a little the training time.
- Modules that introduce **feature integration** (also called **path aggregation**) like skip connection or feature pyramids like **Feature Pyramid Networks (FPN)** or **(SFAM)**.
- Finding good **activation functions**. Many have been proposed to solve the gradient vanish problem (refer to the book). ???
- Post-processing methods like **Non-Maximum Suppresion (NMS)**, where BBoxes that badly predict the same object (redundant or incorrect BBoxes) are filtered out so that the output is a single bounding box for each object in the image. This is no longer required in anchor-free methods. ???


## YOLOv4 (2020)
The main focus of this work is to make the training more **efficient**: they optimize YOLOv3 such that it can provide an high-quality, real-time convining object-detector even when trained on less resources (like single GPU training).

Specifically, they tried to find optimal balance between the input network resolution, the convolutional layer number, the parameter number ($\text{filter\_size}^2 \cdot \text{filters} \cdot \frac{\text{channel}}{\text{groups}}$), and the number of layer outputs ($\text{filters}$).

By experimenting with different backbones they noticed that their performance differed on different datasets. One could perform better on a specific dataset, but worse in a different one. They considered that the ideal case was a network with *larger receptive field size* and a *larger number of parameters*, so they chose a variant of **DarkNet (CSPNet) as backbone**.

In general more parameters lead to a greater capacity of the model to detect multiple objects of different sizes in a single image. Also a large receptive field allows viewing the entire object/context.

They added the **SPP** module to enlarge the receptive field, the **PANet path-aggregation as neck** (instead of FPN in YOLOv3) and YOLOv3 as the head of the architecture.

They do NOT use Cross-GPU Batch Normalization or other expensive devices to allow reproducibility.

They also added a new method of data augmentation (Mosaic and Self-Adversarial Training) and modified some of the used method (Spatial Attention Module and PAN) to make the design more efficient.

**Mosaic data augmentation**: it is a method that mixes 4 images. This allows detection of object outside their normal context. In addition, batch normalization calculates activation statistics from 4 different images on each layer. This significantly reduces the need for a large mini-batch size.

For the backbone they used:
- Class label smoothing
- Data augmentation (specifically CutMix and Mosaic Data augmentation)
- DropBlock regularization
- Mish activation function
- Cross-stage partial connections (Skip-connections)
- Multi-input weighted residual connections (MiWRC)

For the head they used:
- CIoU loss
- Cross mini-Batch Normalization
- Mosaic data augemtation and Self-Adversarial Training
- Eliminate grid sensitivity ???
- Used multiple anchors for a single ground truth ???
- Cosine annealing scheduler
- Search optimal hyperparameters with genetic algorithms
- Random training shapes
- Mish activation
- DIoU-NMS
- SPP block, SAM block, PAN block (neck)

They provide all the hyperparameters they used. They used ImageNet for classification and MS COCO for object detection, using different hyperparameters for each.

They experimented many parameters and features.

## YOLOv5 (2020)
YOLOv5 has a modular structure, allowing for easy customisation, which is an important reason for its popularity.

YOLOv5 uses a **Cross-Stage Partial (CSP)** Net (CSPDarknet53) as the **backbone**.

It uses a **spatial pyramid pooling - fast (SPPF)** module (allowing feature extraction at various scales) and a **Path Aggregation Network (PAN)** module as the **neck** for effective multi-scale feature fusion, along with additional up-sampling layers to improve the resolution of feature maps.
The **head** of YOLOv5 consists of a sequence of **convolutional layers** that generate predictions for bounding boxes and class labels, specifically YOLOv3. YOLOv5 utilises **anchor-based predictions**, linking each bounding box with a set of predefined anchor boxes of specific shapes and sizes. 

As activation function it uses Leaky ReLu. The loss function calculation involves two main components: **binary cross-entropy** (for computing class and objectness losses) and **Complete Intersection over Union (CIoU)** (for localization accuracy).

YOLOv5 incorporates **mosaic augmentation** and **CutOut** to enhance its ability to generalize to new situations. It uses the **Non-Maximal Suppression (NMS)** post-processing technique.

## YOLOv6 (2022)
New version that uses a different network (EfficientNet-L2) as backbone.

It also introduces a new method for generating the anchor boxes, called "dense anchor boxes". Designed for efficiency since its main focus was industrial application.

## YOLOv7 (2022)
It introduced techniques like Model Scaling Techniques, Re-parameterization Planning and
Coarse-to-fine auxiliary head supervision to improve its efficiency.

## YOLOv8 (2023)
The main improvement of YOLOv8 was introducing anchor-free detection, which simplified the architecture and improved the detection of small objects.

The **backbone** is made by an enhanced version of the **CSPDarknet** which contains multiple **Cross Stage Partial (CSP)** block. Each block contains a split operation (part of the feature map is passed to convolutional layers and the other is directly concatenated to the output of the block), a dense block, a transition layer and a concatenation operation.

It uses advanced activation functions like **SiLU (Swish)** improving gradient flow and feature expressiveness.
It uses task-specific losses (the model provides for multiple tasks like detection, segmentation or classification) and **Mosaic augmentation**, **Cutout** and **MixUp**.

The **neck** is made by a **Path Aggregation Network (PANet)**, this network is build upon *Feature pyramid networks (FPN)* with an additional bottom-up path. Specifically, it contains a bottom-up path for feature extraction, a top-down path for semantic feature propagation, and an additional bottom-up path for further feature hierarchy enhancement. At each level, features from corresponding bottom-up and top-down path are fused usually through element-wise addition or concatenation. It also introduces adaptive feature pooling to enhance multi-scale feature fusion, pooling features from all levels for each Region of Interest (RoI). This design improves the network’s ability to detect objects at various scales, essential to imrpove detection on small objects.

The **head** is an **anchor-free detection head**. This simplifies the model architecture and reduces computational overhead, leading to faster inference times. The anchor-free approach also improves the model’s ability to detect small and densely packed objects.

In all three components it is also used the **C2f (Cross Stage Partial with two fusion)** building block which enhances feature extraction and fusion.

It uses some post-processing techniques:
- Improved **Non-Maximum Suppression (NMS)**: enhanced version that reduces the number of false positives and improves the precision of object detection. how??

To enhance training efficiency YOLOv8 uses:
- **mixed-precision training** that combines 16-bit and 32-bit floating-point operations.
- **automated hyperparameter optimization** where multiple training experiments are run with different hyperparameters settings. 

YOLOv8’s shift to an anchor-free detection head and the introduction of task-specific heads expanded themodel’s versatility, allowing it to handle a wider range of computer vision tasks.

### YOLOv8s (small)

On COCO dataset the YOLOv8s obtains 44.9% Average Precision and 28.6 billion Floating-point Operations per Second (FLOPS).

(Fun fact! Yolov10 was introduced in 2024 and makes use of NMS-free training, making it particularly efficient for edge devices with limited resources)

# Why did we choose YOLOv8?
*"Although there are more than ten versions of YOLO, YOLOv5, YOLOv8, and YOLOv10 have gained particular prominence in edge deployment scenarios. These three variants stand out due to their optimal balance of speed, accuracy, and efficiency, making them especially suitable for resource-constrained environments."*
Reasons:
- Optimized Performance, good balance between efficiency and accuracy
- Scalability, each variant provides multiple sub-variants with respect to architectural depth (nano, small, etc.)
- Ease of Deployment, YOLOv8’s unified API significantly simplify the deployment process
- Continuous Improvement over the previous versions
- Community Support, strong community backing and regular updates


# References
- Ren et al. "Faster r-cnn: towards realtime object detection with region proposal networks." (2015)
- He et al. "Spatial pyramid pooling in deep convolutional networks for visual recognition." (2015)
- Redmon et al. "You only look once: unified, real-time object detection." (2016)
- Lin et al. "Feature pyramid networks for object detection." (2017)
- Redmon and Farhadi "Yolo9000: better, faster, stronger." (2017)
- Joseph and Ali "Yolov3: an incremental improvement." (2018)
- Bochkovskiy et al. "Yolov4: optimal speed and accuracy of object detection." (2020)
- Chuyi et al "YOLOv6: ASingle-Stage Object Detection Framework for Industrial Applications" (2022)
- Chien-Yao et al. "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors" (2022)
- Hussain "YOLOv5, YOLOv8 and YOLOv10: the go-to detectors for real-time vision." (2024)