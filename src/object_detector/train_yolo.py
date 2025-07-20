#!/usr/bin/env python3
"""
Configurable YOLOv8 training script for single-class or multi-class fish detection.
Supports switching between modes with minimal configuration changes.
"""

import os
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train YOLOv8 for Fish Detection')
    
    parser.add_argument('--mode', type=str, choices=['single', 'multi'], default='single',
                       help='Training mode: single-class or multi-class (default: single)')
    parser.add_argument('--model_size', type=str, choices=['n', 's', 'm', 'l', 'x'], default='s',
                       help='YOLO model size (default: s)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training (default: 16)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size (default: 640)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: auto, cpu, cuda (default: auto)')
    parser.add_argument('--project', type=str, default='runs/train',
                       help='Project directory for saving results')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name (auto-generated if not specified)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint path')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained weights (uses COCO pretrained if not specified)')
    
    return parser.parse_args()

def get_config_path(mode):
    """Get the configuration file path based on training mode"""
    config_dir = Path(__file__).parent / 'configs'
    if mode == 'single':
        return config_dir / 'single_class_config.yaml'
    else:
        return config_dir / 'multi_class_config.yaml'

def load_config(config_path):
    """Load and validate configuration file"""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate paths exist
    for split in ['train', 'val', 'test']:
        if split in config:
            path = Path(config[split])
            if not path.exists():
                raise FileNotFoundError(f"{split} path does not exist: {path}")
    
    return config

def detect_device(requested_device):
    """Detect and validate the best available device"""
    if requested_device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = requested_device
    
    # Validate device
    if device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("⚠️  MPS not available, falling back to CPU")
        device = 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = 'cpu'
    
    return device

def train_model(args):
    """Train YOLOv8 model with specified configuration"""
    
    # Set up configuration
    config_path = get_config_path(args.mode)
    config = load_config(config_path)
    
    # Detect and set device
    device = detect_device(args.device)
    
    # Set experiment name if not provided
    if args.name is None:
        args.name = f'deepfish_{args.mode}_class_{args.model_size}'
    
    # Initialize model
    if args.resume:
        print(f"📁 Resuming training from: {args.resume}")
        model = YOLO(args.resume)
    elif args.pretrained:
        print(f"📁 Loading pretrained weights from: {args.pretrained}")
        model = YOLO(args.pretrained)
    else:
        print(f"📁 Loading COCO pretrained YOLOv8{args.model_size}")
        model = YOLO(f"yolov8{args.model_size}.pt")
    
    print(f"🏋️  Training Configuration:")
    print(f"   Mode: {args.mode}-class")
    print(f"   Model: YOLOv8{args.model_size}")
    print(f"   Classes: {config['nc']}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Image size: {args.imgsz}")
    print(f"   Device: {device}")
    print(f"   Config: {config_path}")
    
    # Training parameters
    train_params = {
        'data': str(config_path),
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch_size,
        'device': device,
        'project': args.project,
        'name': args.name,
        'save': True,
        'plots': True,
        'patience': 30,
        'workers': 4,
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'save_period': 10,
        'val': True,
        'rect': False,  # Rectangular training disabled for better compatibility
        'cos_lr': True,  # Cosine learning rate scheduler
        'close_mosaic': 10,  # Disable mosaic augmentation in last N epochs
    }
    
    # Mode-specific parameters
    if args.mode == 'single':
        # Single-class specific optimizations
        train_params.update({
            'single_cls': True,  # KEY PARAMETER: Treat multi-class dataset as single class
            'cls': 0.5,          # Classification loss weight
            'box': 7.5,          # Box regression loss weight  
            'dfl': 1.5,          # Distribution focal loss weight
            'augment': True,
            'mixup': 0.1,
            'copy_paste': 0.1,
        })
    else:
        # Multi-class specific optimizations  
        train_params.update({
            'single_cls': False,  # Multi-class mode
            'cls': 1.0,           # Higher classification loss for multi-class
            'box': 7.5,
            'dfl': 1.5,
            'augment': True,
            'mixup': 0.15,        # Higher mixup for multi-class
            'copy_paste': 0.3,    # Higher copy-paste for multi-class
        })
    
    print(f"\n🚀 Starting training...")
    
    # Train the model
    results = model.train(**train_params)
    
    print(f"\n✅ Training completed!")
    print(f"📊 Results saved to: {results.save_dir}")
    
    return model, results

def validate_model(model, config_path, device='auto'):
    """Validate the trained model"""
    print(f"\n🔍 Validating model...")
    
    # Use the same device detection logic
    device = detect_device(device)
    
    results = model.val(
        data=str(config_path),
        split='val',
        device=device,
        save=True,
        plots=True,
        conf=0.25,
        iou=0.6,
        max_det=300
    )
    
    print(f"📊 Validation Results:")
    print(f"   mAP@0.5: {results.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"   Precision: {results.box.mp:.4f}")
    print(f"   Recall: {results.box.mr:.4f}")
    
    return results

def test_model(model, config_path, device='auto'):
    """Test the trained model"""
    print(f"\n🧪 Testing model...")
    
    # Use the same device detection logic
    device = detect_device(device)
    
    results = model.val(
        data=str(config_path),
        split='test',
        device=device,
        save=True,
        plots=True,
        conf=0.25,
        iou=0.6,
        max_det=300
    )
    
    print(f"📊 Test Results:")
    print(f"   mAP@0.5: {results.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"   Precision: {results.box.mp:.4f}")
    print(f"   Recall: {results.box.mr:.4f}")
    
    return results

def main():
    """Main training pipeline"""
    args = parse_arguments()
    
    # Print device availability
    print(f"🔧 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔧 CUDA devices: {torch.cuda.device_count()}")
        print(f"🔧 Current device: {torch.cuda.current_device()}")
    
    print(f"🔧 MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    
    # Test device detection
    detected_device = detect_device(args.device)
    print(f"🔧 Detected device: {detected_device}")
    
    # Train model
    model, train_results = train_model(args)
    
    # Get config path for evaluation
    config_path = get_config_path(args.mode)
    
    # Validate model
    val_results = validate_model(model, config_path, args.device)
    
    # Test model  
    test_results = test_model(model, config_path, args.device)
    
    # Save final model
    output_dir = Path('saved_models')
    output_dir.mkdir(exist_ok=True)
    
    model_name = f"yolov8{args.model_size}_{args.mode}_class_fish.pt"
    model_path = output_dir / model_name
    
    # Copy best weights to saved_models
    best_weights = train_results.save_dir / 'weights' / 'best.pt'
    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, model_path)
        print(f"💾 Model saved to: {model_path}")
    
    print(f"\n🎉 Training pipeline completed!")
    print(f"📊 Final Performance:")
    print(f"   Mode: {args.mode}-class")
    print(f"   Validation mAP@0.5: {val_results.box.map50:.4f}")
    print(f"   Test mAP@0.5: {test_results.box.map50:.4f}")

if __name__ == "__main__":
    main()
