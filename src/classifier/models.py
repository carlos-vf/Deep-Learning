# src/utils/models.py

import torch
from pathlib import Path
from classifier.classification import FishCNN

def load_species_classifier(model_path, device):
    """
    Loads your custom pre-trained FishCNN species classifier.
    It reads the number of classes and class names from the checkpoint file.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"CNN model checkpoint not found at: {model_path}")

    # Load the checkpoint dictionary
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get necessary info from the checkpoint
    num_classes = checkpoint.get('num_classes')
    class_names = checkpoint.get('class_names')

    if num_classes is None or class_names is None:
        raise KeyError("Checkpoint must contain 'num_classes' and 'class_names' keys.")

    # 1. Build the model skeleton with the correct number of classes
    model = FishCNN(num_classes).to(device)
    
    # 2. Load the saved weights into the skeleton
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 3. Set the model to evaluation mode
    model.eval()
        
    # Return the model and the list of species names
    return model, class_names
