import torchvision.transforms as transforms

# Transformations for TRAINING the CNN.
# Includes data augmentation to make the model more robust.
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Transformations for INFERENCE (validation, testing, and the final pipeline).
# No augmentation is used here to ensure consistent, real-world evaluation.
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
