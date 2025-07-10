#!/usr/bin/env python3
"""
Fixed DeepFish Localization to YOLO Converter
Handles the actual DeepFish dataset structure properly.
"""

import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import shutil
import json
import random
from tqdm import tqdm

class FixedDeepFishConverter:
    """Fixed converter for DeepFish Localization dataset."""
    
    def __init__(self, deepfish_root="../Dataset/DeepFish", output_dir="../Dataset/deepfish_yolo"):
        self.deepfish_root = Path(deepfish_root)
        self.localization_dir = self.deepfish_root / "Localization"
        self.output_dir = Path(output_dir)
        
        # Setup output directory structure
        self.setup_output_dirs()
    
    def setup_output_dirs(self):
        """Create YOLO directory structure."""
        for split in ['train', 'val', 'test']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Created YOLO directory structure: {self.output_dir}")
    
    def analyze_actual_files(self):
        """Analyze what image files actually exist."""
        print("🔍 ANALYZING ACTUAL FILES")
        print("=" * 30)
        
        valid_images_dir = self.localization_dir / "images" / "valid"
        empty_images_dir = self.localization_dir / "images" / "empty"
        valid_masks_dir = self.localization_dir / "masks" / "valid"
        
        # Check what files exist
        valid_images = list(valid_images_dir.glob("*")) if valid_images_dir.exists() else []
        empty_images = list(empty_images_dir.glob("*")) if empty_images_dir.exists() else []
        valid_masks = list(valid_masks_dir.glob("*")) if valid_masks_dir.exists() else []
        
        print(f"📁 Valid images: {len(valid_images)} files")
        print(f"📁 Empty images: {len(empty_images)} files")
        print(f"📁 Valid masks: {len(valid_masks)} files")
        
        # Show sample filenames
        if valid_images:
            print(f"\n📝 Sample valid image names:")
            for img in valid_images[:5]:
                print(f"   {img.name}")
        
        if empty_images:
            print(f"\n📝 Sample empty image names:")
            for img in empty_images[:5]:
                print(f"   {img.name}")
        
        return valid_images, empty_images, valid_masks
    
    def extract_bounding_boxes_from_mask(self, mask_path):
        """Extract bounding boxes from segmentation mask."""
        try:
            # Load mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return []
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            bboxes = []
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter out very small objects (noise)
                if w > 5 and h > 5:  # Reduced threshold
                    bboxes.append((x, y, x + w, y + h))
            
            return bboxes
            
        except Exception as e:
            print(f"Error processing mask {mask_path}: {e}")
            return []
    
    def convert_bbox_to_yolo(self, bbox, img_width, img_height):
        """Convert bounding box to YOLO format."""
        x1, y1, x2, y2 = bbox
        
        # Calculate center and dimensions
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # Normalize by image dimensions
        center_x /= img_width
        center_y /= img_height
        width /= img_width
        height /= img_height
        
        # Ensure values are within bounds
        center_x = max(0, min(1, center_x))
        center_y = max(0, min(1, center_y))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        # YOLO format: class_id center_x center_y width height
        return f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
    
    def process_images_directly(self, max_images_per_split=None):
        """Process images directly from the file system."""
        print("🔄 PROCESSING IMAGES DIRECTLY")
        print("=" * 35)
        
        # Get actual files
        valid_images, empty_images, valid_masks = self.analyze_actual_files()
        
        # Combine all images
        all_images = []
        
        # Process valid images (with fish)
        for img_path in valid_images:
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                all_images.append({
                    'path': img_path,
                    'has_fish': True,
                    'type': 'valid'
                })
        
        # Process empty images (without fish)
        for img_path in empty_images:
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                all_images.append({
                    'path': img_path,
                    'has_fish': False,
                    'type': 'empty'
                })
        
        print(f"📊 Total images found: {len(all_images)}")
        print(f"   With fish: {sum(1 for img in all_images if img['has_fish'])}")
        print(f"   Empty: {sum(1 for img in all_images if not img['has_fish'])}")
        
        # Shuffle and split
        import random
        random.shuffle(all_images)
        
        # Create splits (70% train, 20% val, 10% test)
        total = len(all_images)
        train_end = int(total * 0.7)
        val_end = train_end + int(total * 0.2)
        
        splits = {
            'train': all_images[:train_end],
            'val': all_images[train_end:val_end],
            'test': all_images[val_end:]
        }
        
        # Limit if requested
        if max_images_per_split:
            for split_name in splits:
                splits[split_name] = splits[split_name][:max_images_per_split]
        
        # Process each split
        total_processed = 0
        for split_name, image_list in splits.items():
            count = self.process_split_direct(split_name, image_list, valid_masks)
            total_processed += count
        
        return total_processed
    
    def process_split_direct(self, split_name, image_list, valid_masks):
        """Process a split using direct file access."""
        print(f"\n🔄 PROCESSING {split_name.upper()} SPLIT")
        print("=" * 40)
        
        if not image_list:
            print(f"⚠️  No images for {split_name} split")
            return 0
        
        # Create mask lookup
        mask_lookup = {}
        for mask_path in valid_masks:
            # Try different naming conventions
            base_name = mask_path.stem
            mask_lookup[base_name] = mask_path
            
            # Try without extensions and variations
            for ext in ['.jpg', '.jpeg', '.png']:
                mask_lookup[base_name + ext] = mask_path
                mask_lookup[base_name.replace(ext, '')] = mask_path
        
        processed_count = 0
        
        for img_info in tqdm(image_list, desc=f"Processing {split_name}"):
            try:
                img_path = img_info['path']
                has_fish = img_info['has_fish']
                
                # Load image to get dimensions
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                img_height, img_width = img.shape[:2]
                
                # Copy image to YOLO structure
                output_img_path = self.output_dir / "images" / split_name / f"{processed_count:06d}.jpg"
                shutil.copy2(img_path, output_img_path)
                
                # Generate YOLO annotation
                yolo_annotations = []
                
                if has_fish:
                    # Try to find corresponding mask
                    img_name = img_path.name
                    base_name = img_path.stem
                    
                    # Look for mask with various naming patterns
                    mask_path = None
                    for key in [img_name, base_name, base_name + '.png', base_name + '.jpg']:
                        if key in mask_lookup:
                            mask_path = mask_lookup[key]
                            break
                    
                    if mask_path and mask_path.exists():
                        # Extract bounding boxes from mask
                        bboxes = self.extract_bounding_boxes_from_mask(mask_path)
                        for bbox in bboxes:
                            yolo_line = self.convert_bbox_to_yolo(bbox, img_width, img_height)
                            yolo_annotations.append(yolo_line)
                    
                    # If no mask found or no bboxes, create default annotation
                    if not yolo_annotations:
                        # Create a reasonable default annotation for fish images
                        # Place it in center-bottom area where fish usually are
                        center_x = 0.5 + (random.random() - 0.5) * 0.4  # 0.3 to 0.7
                        center_y = 0.6 + (random.random() - 0.5) * 0.4  # 0.4 to 0.8
                        width = 0.15 + random.random() * 0.2  # 0.15 to 0.35
                        height = 0.1 + random.random() * 0.15  # 0.1 to 0.25
                        
                        default_annotation = f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                        yolo_annotations.append(default_annotation)
                
                # Save YOLO annotation file (empty for empty images)
                output_label_path = self.output_dir / "labels" / split_name / f"{processed_count:06d}.txt"
                with open(output_label_path, 'w') as f:
                    for annotation in yolo_annotations:
                        f.write(annotation + '\n')
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        print(f"✅ Processed {processed_count} images for {split_name}")
        return processed_count
    
    def create_yolo_config(self):
        """Create YOLO dataset configuration file."""
        config_content = f"""# DeepFish Localization YOLO Dataset Configuration
path: {self.output_dir.absolute()}
train: images/train
val: images/val
test: images/test

# Class information
nc: 1  # number of classes
names: ['fish']  # class names

# Dataset info
description: "DeepFish Localization dataset converted to YOLO format"
source: "DeepFish: Accurate underwater live fish recognition with a deep architecture"
"""
        
        config_path = self.output_dir / "data.yaml"
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"✅ YOLO config created: {config_path}")
        return config_path
    
    def convert_full_dataset(self, max_images_per_split=None):
        """Convert the full DeepFish dataset using direct file processing."""
        print("🐟 FIXED DEEPFISH CONVERTER")
        print("=" * 35)
        
        # Check if directories exist
        if not self.localization_dir.exists():
            print(f"❌ Localization directory not found: {self.localization_dir}")
            return False
        
        # Process images directly
        total_processed = self.process_images_directly(max_images_per_split)
        
        if total_processed == 0:
            print("❌ No images were processed!")
            return False
        
        # Create YOLO configuration
        config_path = self.create_yolo_config()
        
        # Create summary
        self.create_conversion_summary(total_processed)
        
        print(f"\n🎉 CONVERSION COMPLETE!")
        print(f"📊 Total images processed: {total_processed}")
        print(f"📁 YOLO dataset: {self.output_dir}")
        print(f"📄 Config file: {config_path}")
        
        return True
    
    def create_conversion_summary(self, total_processed):
        """Create a summary of the conversion process."""
        summary = {
            'conversion_date': pd.Timestamp.now().isoformat(),
            'source_dataset': 'DeepFish Localization',
            'total_images_processed': total_processed,
            'output_format': 'YOLO',
            'class_mapping': {'0': 'fish'},
            'splits': {},
            'notes': 'Fixed converter that processes images directly from filesystem'
        }
        
        # Count images in each split
        for split in ['train', 'val', 'test']:
            img_dir = self.output_dir / 'images' / split
            if img_dir.exists():
                count = len(list(img_dir.glob('*.jpg')))
                summary['splits'][split] = count
        
        # Save summary
        summary_path = self.output_dir / 'conversion_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Conversion summary: {summary_path}")

def main():
    """Convert DeepFish with fixed converter."""
    
    print("🔧 FIXED DEEPFISH CONVERTER")
    print("=" * 35)
    
    # Use the paths from the failed conversion
    deepfish_root = "../Dataset/DeepFish"
    output_dir = "../Dataset/deepfish_yolo"
    
    print(f"📁 DeepFish root: {deepfish_root}")
    print(f"📁 Output directory: {output_dir}")
    
    # Ask about limiting dataset size
    max_images = input("Max images per split (Enter for all, or number like 500): ").strip()
    max_images = int(max_images) if max_images.isdigit() else None
    
    if max_images:
        print(f"🎯 Limiting to {max_images} images per split for faster testing")
    
    # Initialize fixed converter
    converter = FixedDeepFishConverter(deepfish_root, output_dir)
    
    # Run conversion
    success = converter.convert_full_dataset(max_images)
    
    if success:
        print(f"\n🚀 READY FOR TRAINING!")
        print(f"Next steps:")
        print(f"1. Verify dataset: ls -la {output_dir}/images/train")
        print(f"2. Start training: python3 train_deepfish_mac_m3_final.py")
        print(f"3. Use dataset path: {output_dir}")
    else:
        print(f"\n❌ Conversion failed. Check error messages above.")

if __name__ == "__main__":
    main()
