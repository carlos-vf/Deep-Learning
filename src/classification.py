import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

def count_species(dataset_path):
    """Count the number of species by counting folders in the train dataset"""
    train_path = Path(dataset_path) / 'train'
    
    if not train_path.exists():
        raise FileNotFoundError(f"Train dataset not found at: {train_path}")
    
    # Count directories (species) in train folder
    species_dirs = [d for d in train_path.iterdir() if d.is_dir()]
    species_count = len(species_dirs)
    species_names = [d.name for d in species_dirs]
    
    print(f"Found {species_count} species in {train_path}")
    print(f"Species: {species_names}")
    
    return species_count, species_names

def load_dataset(dataset_path, batch_size=32):
    """Load dataset from given path and return data loaders"""
    # Check if dataset exists
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    
    # Check if train, val, test folders exist
    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split
        if not split_path.exists():
            raise FileNotFoundError(f"{split} dataset not found at: {split_path}")
    
    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(str(dataset_path / 'train'), transform=train_transform)
    test_dataset = datasets.ImageFolder(str(dataset_path / 'test'), transform=test_transform)
    val_dataset = datasets.ImageFolder(str(dataset_path / 'val'), transform=test_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Get number of classes
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {train_dataset.classes}")
    
    return train_loader, test_loader, val_loader, num_classes, train_dataset.classes

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Fish Classification CNN')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset directory (should contain train/val/test folders)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--model_name', type=str, default='fish_cnn_model',
                        help='Name for saved model (default: fish_cnn_model)')
    
    return parser.parse_args()

# Enable MPS for Mac GPU acceleration
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define CNN model
class FishCNN(nn.Module):
    def __init__(self, num_classes):
        super(FishCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 1024)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1024, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Convolutional layers
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 128 * 16 * 16)
        
        # Fully connected layers
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.dropout3(self.relu(self.fc3(x)))
        x = self.fc4(x)
        
        return x

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Count species in the dataset
    num_classes, class_names = count_species(args.dataset)
    
    # Load dataset
    train_loader, test_loader, val_loader, num_classes_verify, class_names_verify = load_dataset(
        args.dataset, args.batch_size
    )
    
    # Verify that species count matches
    assert num_classes == num_classes_verify, f"Species count mismatch: {num_classes} vs {num_classes_verify}"
    
    # Initialize model
    model = FishCNN(num_classes).to(device)
    print(f"Model moved to device: {device}")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train the model
    print("Starting training...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=args.epochs
    )

    # Save the trained model
    os.makedirs('saved_models', exist_ok=True)
    model_path = f'saved_models/{args.model_name}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_classes': num_classes,
        'class_names': class_names_verify,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'args': vars(args)
    }, model_path)

    print(f"Model saved to '{model_path}'")
    print(f"Classes: {class_names_verify}")

    # Evaluate on test set
    print("\nEvaluating model on test set...")
    test_accuracy, correct_predictions, total_predictions = evaluate_model(model, test_loader, device)

    print(f"\nTest Results:")
    print(f"Correct predictions: {correct_predictions}/{total_predictions}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Plot training results
    plot_training_results(train_losses, val_losses, train_accs, val_accs)

def plot_training_results(train_losses, val_losses, train_accs, val_accs):
    """Plot training and validation results"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.show()

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set and return accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy, correct, total

if __name__ == "__main__":
    main()
