#!/usr/bin/env python3
"""
Script to convert Deepfish dataset to YOLO format for object detection.

Current Deepfish structure:
- Multiple numbered directories (7117, 7268, etc.)
- Each directory contains train/ and valid/ subdirectories
- Images and labels are in the same directory
- Additional test/ directory at root level
- Labels are already in YOLO format

Target YOLO structure:
- train/images/ and train/labels/
- val/images/ and val/labels/
- test/images/ and test/labels/
- data.yaml configuration file
"""

import os
import shutil
import json
from pathlib import Path
import argparse


def collect_files(source_dir):
    """Collect all image and label files from nested directories."""
    image_files = []
    label_files = []
    
    source_path = Path(source_dir)
    
    # Find all JPG files recursively
    for img_file in source_path.rglob("*.jpg"):
        # Skip if in Nagative_samples directory
        if "Nagative_samples" in str(img_file):
            continue
            
        # Look for corresponding label file
        label_file = img_file.with_suffix('.txt')
        if label_file.exists():
            image_files.append(img_file)
            label_files.append(label_file)
        else:
            print(f"Warning: No label file found for {img_file}")
    
    return image_files, label_files


def split_data(image_files, label_files, val_ratio=0.2, test_ratio=0.1):
    """Split data into train/val/test sets."""
    import random
    
    # Create paired list and shuffle
    paired_files = list(zip(image_files, label_files))
    random.shuffle(paired_files)
    
    total_files = len(paired_files)
    test_size = int(total_files * test_ratio)
    val_size = int(total_files * val_ratio)
    train_size = total_files - test_size - val_size
    
    print(f"Dataset split:")
    print(f"  Total files: {total_files}")
    print(f"  Train: {train_size}")
    print(f"  Validation: {val_size}")
    print(f"  Test: {test_size}")
    
    # Split the data
    test_files = paired_files[:test_size]
    val_files = paired_files[test_size:test_size + val_size]
    train_files = paired_files[test_size + val_size:]
    
    return train_files, val_files, test_files


def copy_files(files, dest_dir, split_name):
    """Copy files to destination directory."""
    images_dir = dest_dir / split_name / "images"
    labels_dir = dest_dir / split_name / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Copying {len(files)} files to {split_name}...")
    
    for i, (img_file, label_file) in enumerate(files):
        if i % 100 == 0:  # Progress indicator
            print(f"  {i}/{len(files)} files copied...")
        
        # Copy image
        dest_img = images_dir / img_file.name
        shutil.copy2(img_file, dest_img)
        
        # Copy label
        dest_label = labels_dir / label_file.name
        shutil.copy2(label_file, dest_label)


def create_data_yaml(output_dir, num_classes=1, class_names=None):
    """Create data.yaml configuration file."""
    if class_names is None:
        class_names = ['Fish']
    
    data_config = {
        'train': str(output_dir / 'train' / 'images'),
        'val': str(output_dir / 'val' / 'images'),
        'test': str(output_dir / 'test' / 'images'),
        'nc': num_classes,
        'names': class_names
    }
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"train: {data_config['train']}\n")
        f.write(f"val: {data_config['val']}\n")
        f.write(f"test: {data_config['test']}\n")
        f.write(f"nc: {data_config['nc']}\n")
        f.write(f"names: {data_config['names']}\n")
    
    print(f"Created data.yaml at {yaml_path}")
    return yaml_path


def convert_deepfish_to_yolo(source_dir, output_dir, val_ratio=0.2, test_ratio=0.1):
    """
    Convert Deepfish dataset to YOLO format.
    
    Args:
        source_dir: Path to Deepfish dataset directory
        output_dir: Path to output YOLO dataset directory
        val_ratio: Ratio of validation data (default: 0.2)
        test_ratio: Ratio of test data (default: 0.1)
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    print(f"Converting Deepfish dataset from {source_path} to YOLO format at {output_path}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all files
    print("Collecting image and label files...")
    image_files, label_files = collect_files(source_path)
    
    if len(image_files) == 0:
        raise ValueError("No image files found in source directory")
    
    # Split data
    train_files, val_files, test_files = split_data(image_files, label_files, val_ratio, test_ratio)
    
    # Copy files to destination
    copy_files(train_files, output_path, "train")
    copy_files(val_files, output_path, "val")
    copy_files(test_files, output_path, "test")
    
    # Read class names from classes.txt if it exists
    classes_file = source_path / "classes.txt"
    class_names = ['Fish']  # default
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
    
    # Create data.yaml
    create_data_yaml(output_path, len(class_names), class_names)
    
    print(f"\n✅ Dataset conversion completed!")
    print(f"📁 Output directory: {output_path}")
    print(f"📊 Total files processed: {len(image_files)}")
    print(f"📝 Classes: {class_names}")


def main():
    parser = argparse.ArgumentParser(description='Convert Deepfish dataset to YOLO format')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to Deepfish dataset directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output YOLO dataset directory')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Ratio of validation data (default: 0.2)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Ratio of test data (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible splits (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducible results
    import random
    random.seed(args.seed)
    
    # Validate arguments
    if not Path(args.source).exists():
        raise FileNotFoundError(f"Source directory not found: {args.source}")
    
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1.0")
    
    # Convert dataset
    convert_deepfish_to_yolo(args.source, args.output, args.val_ratio, args.test_ratio)


if __name__ == "__main__":
    main()