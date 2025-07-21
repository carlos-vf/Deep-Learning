import yaml
from pathlib import Path
from tqdm import tqdm
import argparse

def load_class_mapping(config_path):
    """Load class mapping from the specified config file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create a mapping from class name (lowercase for matching) to its index
    name_to_index = {name.lower().replace(" ", "_"): idx for idx, name in enumerate(config['names'])}
    
    print("🔧 Class mapping loaded:")
    for name, idx in name_to_index.items():
        print(f"   {idx}: {name}")
    
    return name_to_index

def extract_species_from_filename(filename, known_species):
    """
    Extracts a species name from a filename by finding the best match from a known list.
    
    Args:
        filename (str): The name of the image file.
        known_species (list): A list of known species names (lowercase, with underscores).
    
    Returns:
        str: The matched species name, or None if no match is found.
    """
    # Prepare the filename for matching (lowercase, with underscores)
    name_lower = filename.lower().replace('.jpg', '').replace('.txt', '')
    
    # Sort known species by length (descending) to match longer names first
    # e.g., match 'caranx_sexfasciatus' before 'caranx'
    sorted_species = sorted(known_species, key=len, reverse=True)
    
    for species_pattern in sorted_species:
        if species_pattern in name_lower:
            return species_pattern
            
    return None

def fix_labels_for_split(split_name, dataset_dir, name_to_index):
    """Fix all labels for a given dataset split with correct class indices."""
    
    labels_dir = dataset_dir / split_name / "labels"
    
    if not labels_dir.exists():
        print(f"⚠️  Warning: Label directory for '{split_name}' not found at {labels_dir}. Skipping.")
        return

    print(f"\n🔍 Processing {split_name} labels in: {labels_dir}")
    
    fixed_count = 0
    error_count = 0
    
    label_files = list(labels_dir.glob("*.txt"))
    
    # Use tqdm for a nice progress bar
    for label_file in tqdm(label_files, desc=f"Fixing {split_name} labels"):
        try:
            # Extract species from the label file's name
            species = extract_species_from_filename(label_file.name, list(name_to_index.keys()))
            
            if species is None:
                # This warning is now more meaningful as it means no known species was found
                # print(f"⚠️  Could not extract a known species from: {label_file.name}")
                error_count += 1
                continue
            
            # Get the correct class index from our mapping
            correct_class_idx = name_to_index[species]
            
            # Read the label file
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            updated_lines = []
            file_was_changed = False
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    # Check if the label is already correct
                    if int(parts[0]) != correct_class_idx:
                        parts[0] = str(correct_class_idx)
                        file_was_changed = True
                    updated_lines.append(' '.join(parts) + '\n')
            
            # Write the updated lines back to the file only if changes were made
            if file_was_changed:
                with open(label_file, 'w') as f:
                    f.writelines(updated_lines)
                fixed_count += 1
                
        except Exception as e:
            print(f"❌ Error processing {label_file.name}: {e}")
            error_count += 1
    
    print(f"\n📊 Results for {split_name}:")
    print(f"   Files checked: {len(label_files)}")
    print(f"   Files updated: {fixed_count}")
    print(f"   Files skipped (no species match): {error_count}")

def main():
    """Main script to fix all dataset labels."""
    parser = argparse.ArgumentParser(description='Fix YOLO dataset labels based on filenames.')
    parser.add_argument('--dataset-dir', type=str, required=True, help='Path to the root of the YOLO dataset directory.')
    parser.add_argument('--config', type=str, default='src/object_detector/configs/deepfish_multi_class_config.yaml', help='Path to the dataset config YAML file.')
    args = parser.parse_args()

    print("🔧 Fixing all dataset labels...\n")
    
    # Load the class mapping once
    name_to_index = load_class_mapping(args.config)
    dataset_dir = Path(args.dataset_dir)
    
    # Fix all splits
    for split in ["train", "val", "test"]:
        fix_labels_for_split(split, dataset_dir, name_to_index)
    
    print("\n✅ All labels checked! Your dataset is ready.")

if __name__ == "__main__":
    main()
