#!/usr/bin/env python3
"""
Training script for single-class fish detection on specified dataset.
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import yaml
import json
import argparse
import torch
import ultralytics
from ultralytics import YOLO
from object_detector.config import TRAIN_TRANSFORMS, INFERENCE_TRANSFORMS

class Detector:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.config_path = self.dataset_path / "data.yaml"

    @staticmethod
    def load_dataset(dataset_path, batch_size=32):
        # Check if dataset exists
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    
        # Check if train, val, test folders exist
        for split in ['train', 'val', 'test']:
            split_path = dataset_path / split
            if not split_path.exists():
                raise FileNotFoundError(f"{split} dataset not found at: {split_path}")
    
        # This function is designed for YOLO training which doesn't use PyTorch DataLoaders
        # YOLO handles data loading internally through the data.yaml configuration
        print(f"Dataset validated at: {dataset_path}")
        return None, None, None

    @staticmethod
    def parse_arguments():
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(description='Train Fish Classification CNN')
        parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset directory (should contain data.yaml)')
        parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training (default: 8)')
        parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train (default: 50)')
        parser.add_argument('--model_size', type=str, default='s',
                        help='YOLO model size (n, s, m, l, x) (default: s)')
        parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for training (default: 640)')
        parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, mps, cuda) (default: auto)')
        parser.add_argument('--model_name', type=str, default='fish_detector',
                        help='Name for saved model (default: fish_detector)')
    
        return parser.parse_args()

    def train(self, model_size='s', epochs=50, imgsz=640, batch_size=8, device='auto'):
        print(f"🏃 Starting training...")
        print(f"📊 Model: YOLOv8{model_size}")
        print(f"📊 Epochs: {epochs}")
        print(f"📊 Image size: {imgsz}")
        print(f"📊 Batch size: {batch_size}")
        print(f"📊 Device: {device}")
        print(f"📊 Dataset: {self.config_path}")
        
        # Initialize model
        model = YOLO(f"yolov8{model_size}.pt")
        
        # Train model
        results = model.train(
            data=str(self.config_path),
            epochs=epochs,
            imgsz=imgsz,
            patience=20,  # Early stopping patience
            save=True,
            device=device,  # 'auto', 'cpu', 'mps', or specific GPU
            workers=4,
            batch=batch_size,
            project='runs/train',
            name='deepfish_single_class',
            optimizer='AdamW',
            lr0=0.001,  # Lower learning rate for better convergence
            weight_decay=0.0005,
            augment=True,
            mixup=0.1,  # Mixup augmentation for better generalization
            copy_paste=0.1,  # Copy-paste augmentation
            plots=True,
            save_period=10,  # Save checkpoint every 10 epochs
            # Single-class specific optimizations
            cls=0.5,  # Classification loss weight
            box=7.5,  # Box regression loss weight
            dfl=1.5   # Distribution focal loss weight
        )
        
        return model, results
    
    def validate(self, model):
        print("🔍 Validating model...")
        
        # Run validation with less aggressive NMS (better recall)
        results = model.val(
            data=str(self.config_path),
            split='val',
            save=True,
            plots=True,
            conf=0.15,  # Lower confidence = more detections
            iou=0.5,    # Less aggressive NMS
            max_det=300 # Allow more detections
        )
        
        print(f"📊 Validation results:")
        print(f"   mAP@0.5: {results.box.map50:.4f}")
        print(f"   mAP@0.5:0.95: {results.box.map:.4f}")
        print(f"   Precision: {results.box.mp:.4f}")
        print(f"   Recall: {results.box.mr:.4f}")
        
        # Single class specific metrics
        if hasattr(results.box, 'maps'):
            print(f"   Fish detection mAP@0.5: {results.box.maps[0]:.4f}")
        
        return results
    
    def test_model(self, model, conf = 0.25):
        print("🧪 Testing model...")
        
        # Run testing
        results = model.val(
            data=str(self.config_path),
            split='test',
            save=True,
            plots=True,
            conf=conf,
            iou=0.7,     
            max_det=100  
        )
        
        print(f"📊 Test results:")
        print(f"   mAP@0.5: {results.box.map50:.4f}")
        print(f"   mAP@0.5:0.95: {results.box.map:.4f}")
        print(f"   Precision: {results.box.mp:.4f}")
        print(f"   Recall: {results.box.mr:.4f}")
        
        return results
    
def main():
    # Parse command line arguments
    args = Detector.parse_arguments()

    # Initialize detector
    detector = Detector(args.dataset)
    print(f'Detector initialized with dataset: {args.dataset}')

    # Validate dataset structure
    Detector.load_dataset(args.dataset, args.batch_size)

    # Train model
    model, train_results = detector.train(
        model_size=args.model_size,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Validate model
    val_results = detector.validate(model)
    
    # Test model
    test_results = detector.test_model(model)
    
    # Save the trained model (YOLO automatically saves best.pt and last.pt)
    os.makedirs('saved_models', exist_ok=True)
    best_model_path = f'runs/train/deepfish_single_class/weights/best.pt'
    saved_model_path = f'saved_models/{args.model_name}.pt'
    
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, saved_model_path)
        print(f"Model saved to: {saved_model_path}")
    else:
        print("Warning: Best model not found at expected location")

    print(f"\n🎉 Training completed!")
    print(f"📊 Final results:")
    print(f"   Validation mAP@0.5: {val_results.box.map50:.4f}")
    print(f"   Test mAP@0.5: {test_results.box.map50:.4f}")

if __name__ == "__main__":
    main()
