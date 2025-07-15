# YOLO architecture and theory
## Introduction
YOLO is a fast object detection architecture based on Convolutional Neural Networks that works extremely fast thanks to its *proposal-free* scheme.

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

## YOLOv1

First, the image is divided in an $S \times S$ grid. If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object.

Each grid cell predicts **B bounding boxes** and **confidence scores** for those boxes. These confidence scores reflect how confident the model is that the box contains an object and also how accurate it thinks the box is that it predicts. A bounding box is described by 4 numbers: the coordinates of the center, its width and height.

Ideally, we want the confidence score to be equal to the ratio of intersection area over union area (**IOU**) between the predicted box and the ground truth. It is easy to see that, if the two boxes overlaps exactly, the intersaction will be equal to the union (ratio = 1) while if the two boxes are completely separated the intersaction, and ratio, will be 0.

Each grid cell also predicts **C conditional class probability** (conditioned on the presence of an object in that grid). 

This means that in total the predictions are contained in a tensor of dimension:
$$S \times S \times (B\cdot 5 + C)$$
(for every grid there are $B$ boxes and $C$ probabilities, and each box has 5 numbers and an object confidence score)

The network was inspired by GoogLeNet model and it contains 24 convolutional layers followed by 2 fully connected layers.

(There is also a fast version that has just less layers)

The input is of a $448 \times 448$ RGB image (double the resolution of a standard $224 \times 224$ RGB image). It is divided in a $7 \times 7$ grid (so the image is divided in $49$ cells of $4096$ pixels each).

First, we have a convolutional layer where the kernel has size $7 \times 7 \times 64$ and stride $2$ and a maxpool layer of size $2$. Since the stride is $2$ the resulting layer has height and width halved, but also since the kernel has multiple channels the final layer has $3 \cdot 64 = 192$ channels. The max pooling also halves the representation, obtaining the layer $112 \times 112 \times 192$.

A similar sequence of layers (convolutional layers that augment the channels and maxpool layers that halve the representation size) follows until the last 2 fully connected layers. The first FC brings the channels to 1 and the last obtains the $7 \times 7 \times 30$ tensor of predictions. 

To train the network, they first pretrain the first 20 convolutional layers followed by a average-pooling layer and a fully connected layer using the ImageNet dataset. Then they convert the model by adding the 4 convolutional layers and the 2 fully connected layers and train the new network with the weights initialized by the pretraining.

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

