#!/usr/bin/env python3
"""
Fix test dataset labels by mapping species names from filenames to correct class indices
"""

import os
import yaml
from pathlib import Path

def load_class_mapping():
    """Load class mapping from config file"""
    config_path = "/Users/christianfaccio/UniTs/First_Year/Deep_Learning/Deep-Learning/src/object_detector/configs/multi_class_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create mapping from class name to index
    name_to_index = {}
    for idx, name in enumerate(config['names']):
        name_to_index[name] = idx
    
    print("🔧 Class mapping loaded:")
    for name, idx in name_to_index.items():
        print(f"   {idx}: {name}")
    
    return name_to_index

def extract_species_from_filename(filename):
    """Extract species name from filename"""
    # Remove extension
    name = filename.replace('.jpg', '').replace('.txt', '')
    
    # Split by underscore and look for species patterns
    parts = name.split('_')
    
    # Common patterns in the filenames
    species_patterns = [
        'Caranx_sexfasciatus',
        'Acanthopagrus_palmaris', 
        'Lutjanus_russellii',
        'acanthopagrus_and_caranx',
        'acanthopagrus_palmaris',
        'Amniataba_caudivittatus',
        'Epinephelus',
        'Gerres',
        'Caranx',
        'gerres_2',
        'gerres',
        'juvenile',
        'palmaris',
        'caudivittatus',
        'EJP'
    ]
    
    # Try to find matching species pattern
    name_lower = name.lower()
    
    # Check for multi-word species first
    for pattern in species_patterns:
        pattern_lower = pattern.lower()
        if pattern_lower in name_lower:
            return pattern
    
    # Check for F1, F2, etc.
    for part in parts:
        if part.startswith('F') and len(part) == 2 and part[1].isdigit():
            return part
    
    return None

def fix_labels_for_split(split_name):
    """Fix all labels for a given dataset split with correct class indices"""
    
    # Load class mapping
    name_to_index = load_class_mapping()
    
    # Paths
    labels_dir = Path(f"/Users/christianfaccio/UniTs/First_Year/Deep_Learning/Deep-Learning/Datasets/Deepfish_YOLO/{split_name}/labels")
    images_dir = Path(f"/Users/christianfaccio/UniTs/First_Year/Deep_Learning/Deep-Learning/Datasets/Deepfish_YOLO/{split_name}/images")
    
    print(f"\n🔍 Processing {split_name} labels in: {labels_dir}")
    
    fixed_count = 0
    error_count = 0
    species_stats = {}
    
    # Process each label file
    for label_file in labels_dir.glob("*.txt"):
        try:
            # Get corresponding image filename
            image_name = label_file.stem + '.jpg'
            image_path = images_dir / image_name
            
            if not image_path.exists():
                print(f"⚠️  Image not found for {label_file.name}")
                continue
            
            # Extract species from filename
            species = extract_species_from_filename(image_name)
            
            if species is None:
                print(f"⚠️  Could not extract species from: {image_name}")
                error_count += 1
                continue
            
            # Get correct class index
            if species not in name_to_index:
                print(f"⚠️  Species '{species}' not found in class mapping for: {image_name}")
                error_count += 1
                continue
            
            correct_class_idx = name_to_index[species]
            
            # Read and update label file
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            updated_lines = []
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    # Replace class index (first element) with correct one
                    parts[0] = str(correct_class_idx)
                    updated_lines.append(' '.join(parts))
            
            # Write updated labels
            with open(label_file, 'w') as f:
                for line in updated_lines:
                    f.write(line + '\n')
            
            # Update statistics
            if species not in species_stats:
                species_stats[species] = 0
            species_stats[species] += len(updated_lines)
            
            fixed_count += 1
            
        except Exception as e:
            print(f"❌ Error processing {label_file.name}: {e}")
            error_count += 1
    
    print(f"\n📊 Results:")
    print(f"   Fixed files: {fixed_count}")
    print(f"   Errors: {error_count}")
    
    print(f"\n📈 Species distribution after fixing:")
    for species, count in sorted(species_stats.items()):
        class_idx = name_to_index[species]
        print(f"   {species} (class {class_idx}): {count} detections")

def fix_test_labels():
    """Fix test labels - wrapper for backwards compatibility"""
    fix_labels_for_split("test")

def main():
    print("🔧 Fixing all dataset labels...\n")
    
    # Fix all splits
    for split in ["train", "val", "test"]:
        print(f"\n{'='*50}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*50}")
        fix_labels_for_split(split)
    
    print("\n✅ All labels fixed! You can now train/validate/test with correct class indices.")

if __name__ == "__main__":
    main()
