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

## YOLO-Fish

They first preprocessed a lot of data, using labelImg tool to automatically label fish images.


Datasets used in literature:
- LifeCLEF 2014/2015 (10/15 species)
- ImageCLEF
- main resource: Fish4Knowledge (videos)
- Rockfish (problematic)
- QUT (problematic)
- DeepFish
- OzFish


References:
- Salman et al. "Fish species classification in unconstrained underwater environments based on deep learning" (2016)
- Siddiqui et al. "Automatic fish species classification in underwater videos: exploiting pre-trained deep neural network models to compensate for limited labelled data" (2018)
- Salman et al. "Automatic fish detection in underwater videos by a deep neural network-based hybrid motion learning system" (2020)
- Li and Cao "Fast Accurate Fish Detection and Recognition of Underwater Images with Fast R-CNN" (2015)
- "YOLO-Fish: A robust fish detection model to detect fish in realistic
underwater environment" (2022)