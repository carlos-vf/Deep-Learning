#!/usr/bin/env python3
"""
YOLOv8 inference script for single-class or multi-class fish detection.
Supports both single images and batch processing.
"""

import os
import argparse
import cv2
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch
import numpy as np
from PIL import Image

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YOLOv8 Fish Detection Inference')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                       help='Input source: image file, directory, or video file')
    parser.add_argument('--output', type=str, default='runs/detect',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.7,
                       help='NMS IoU threshold (default: 0.7)')
    parser.add_argument('--max_det', type=int, default=1000,
                       help='Maximum detections per image (default: 1000)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: auto, cpu, cuda (default: auto)')
    parser.add_argument('--save_txt', action='store_true',
                       help='Save results as txt files')
    parser.add_argument('--save_conf', action='store_true',
                       help='Save confidence scores in txt files')
    parser.add_argument('--save_crop', action='store_true',
                       help='Save cropped detection images')
    parser.add_argument('--line_thickness', type=int, default=3,
                       help='Bounding box line thickness')
    parser.add_argument('--show_labels', action='store_true', default=True,
                       help='Show labels on bounding boxes')
    parser.add_argument('--show_conf', action='store_true', default=True,
                       help='Show confidence scores on bounding boxes')
    parser.add_argument('--name', type=str, default='exp',
                       help='Experiment name for output directory')
    
    return parser.parse_args()

def load_model(model_path, device='auto'):
    """Load the trained YOLOv8 model"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Handle device selection
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    print(f"📁 Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Print model info
    print(f"🔧 Model loaded successfully")
    print(f"🔧 Model device: {device}")
    
    return model

def detect_single_image(model, image_path, args):
    """Run detection on a single image"""
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    print(f"🔍 Processing image: {image_path}")
    
    # Run inference
    results = model(
        source=image_path,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        project=args.output,
        name=args.name,
        line_width=args.line_thickness,
        show_labels=args.show_labels,
        show_conf=args.show_conf,
        exist_ok=True
    )
    
    # Print detection results
    for i, result in enumerate(results):
        boxes = result.boxes
        if boxes is not None:
            num_detections = len(boxes)
            print(f"✅ Found {num_detections} fish detections")
            
            # Print detection details
            for j, box in enumerate(boxes):
                conf = box.conf.item()
                cls = int(box.cls.item())
                class_name = model.names[cls]
                print(f"   Detection {j+1}: {class_name} (confidence: {conf:.3f})")
        else:
            print(f"❌ No fish detected")
    
    return results

def detect_batch(model, source_dir, args):
    """Run detection on a directory of images"""
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in source_path.rglob('*') if f.suffix.lower() in image_extensions]
    
    if not image_files:
        raise ValueError(f"No image files found in directory: {source_dir}")
    
    print(f"🔍 Processing {len(image_files)} images from: {source_dir}")
    
    # Run batch inference
    results = model(
        source=str(source_path),
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        project=args.output,
        name=args.name,
        line_width=args.line_thickness,
        show_labels=args.show_labels,
        show_conf=args.show_conf,
        exist_ok=True
    )
    
    # Summarize results
    total_detections = 0
    images_with_detections = 0
    
    for result in results:
        if result.boxes is not None:
            num_detections = len(result.boxes)
            if num_detections > 0:
                total_detections += num_detections
                images_with_detections += 1
    
    print(f"📊 Batch Processing Summary:")
    print(f"   Total images processed: {len(results)}")
    print(f"   Images with detections: {images_with_detections}")
    print(f"   Total fish detected: {total_detections}")
    print(f"   Average detections per image: {total_detections/len(results):.2f}")
    
    return results

def detect_video(model, video_path, args):
    """Run detection on a video file"""
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    print(f"🎥 Processing video: {video_path}")
    
    # Run video inference
    results = model(
        source=video_path,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=args.output,
        name=args.name,
        line_width=args.line_thickness,
        show_labels=args.show_labels,
        show_conf=args.show_conf,
        exist_ok=True
    )
    
    print(f"✅ Video processing completed")
    
    return results

def get_model_info(model):
    """Get information about the loaded model"""
    print(f"\n🔧 Model Information:")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Number of classes: {len(model.names)}")
    print(f"   Class names: {list(model.names.values())}")
    
    # Determine if it's single-class or multi-class based on class names
    if len(model.names) == 1 and 'Fish' in model.names.values():
        print(f"   Mode: Single-class detection")
    else:
        print(f"   Mode: Multi-class detection")

def main():
    """Main inference pipeline"""
    args = parse_arguments()
    
    # Print CUDA availability
    print(f"🔧 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔧 CUDA devices: {torch.cuda.device_count()}")
    
    # Handle device selection
    device = args.device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    # Load model
    model = load_model(args.model, device)
    
    # Update args.device with resolved device
    args.device = device
    
    # Get model information
    get_model_info(model)
    
    # Determine source type and run appropriate detection
    source_path = Path(args.source)
    
    if source_path.is_file():
        # Check if it's a video file
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        if source_path.suffix.lower() in video_extensions:
            results = detect_video(model, args.source, args)
        else:
            # Assume it's an image file
            results = detect_single_image(model, args.source, args)
    elif source_path.is_dir():
        # Directory of images
        results = detect_batch(model, args.source, args)
    else:
        raise ValueError(f"Invalid source: {args.source}. Must be an image file, video file, or directory.")
    
    # Print final results
    output_dir = Path(args.output) / args.name
    print(f"\n🎉 Inference completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Processing settings:")
    print(f"   Confidence threshold: {args.conf}")
    print(f"   IoU threshold: {args.iou}")
    print(f"   Max detections: {args.max_det}")
    print(f"   Device: {args.device}")

if __name__ == "__main__":
    main()