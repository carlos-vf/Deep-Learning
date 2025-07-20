"""
Script to convert F4K dataset to YOLO format for single-class fish detection.

This version implements a proportional splitting strategy, ensuring that each
split (train, val, test) contains a representative sample of frames from
every video in the source dataset.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import xml.etree.ElementTree as ET
import random

def create_yolo_dataset(base_path, output_path, val_ratio=0.2, test_ratio=0.1):
    """
    Processes the F4K dataset and creates a new single-class dataset in YOLO format
    with proportional frame splitting from each video.
    """
    base_path = Path(base_path)
    output_path = Path(output_path)

    print("--- Starting F4K to Single-Class YOLO Dataset Preparation (Proportional Split) ---")

    # 1. Create the new directory structure
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 2. Create the single-class data.yaml file
    with open(output_path / 'data.yaml', 'w') as f:
        f.write(f"path: {output_path.resolve()}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n\n")
        f.write("nc: 1\n")
        f.write("names: ['fish']\n")
    print(f"✅ Created single-class data.yaml at {output_path / 'data.yaml'}")

    # 3. Find all videos to process
    video_files = sorted(list((base_path / "videos").rglob('*.mp4')))
    if not video_files:
        print(f"❌ Error: No .mp4 video files found recursively in '{base_path}'.")
        return
    print(f"Found {len(video_files)} videos to process.")

    # 4. Process each video individually
    for video_path in tqdm(video_files, desc="Processing Videos"):
        xml_path = base_path / "gt_bounding_boxes" / f"{video_path.stem}.xml"
        if not xml_path.exists():
            print(f"Warning: No XML found for {video_path.name}, skipping.")
            continue

        # Load all annotations for this video into a dictionary
        tree = ET.parse(xml_path)
        root = tree.getroot()
        annotations = {}
        for frame_node in root.findall('frame'):
            frame_id = int(frame_node.attrib['id'])
            frame_annotations = []
            for obj_node in frame_node.findall('object'):
                if 'fish' in obj_node.attrib.get('objectType', ''):
                    contour_element = obj_node.find('contour')
                    if contour_element is not None and contour_element.text is not None:
                        point_pairs = contour_element.text.strip().split(',')
                        points = [list(map(int, pair.split())) for pair in point_pairs if pair]
                        if points:
                            points_np = np.array(points)
                            x1, y1 = points_np.min(axis=0)
                            x2, y2 = points_np.max(axis=0)
                            frame_annotations.append((x1, y1, x2, y2))
            if frame_annotations:
                annotations[frame_id] = frame_annotations

        # Proportional split of frames WITHIN this video
        annotated_frame_ids = sorted(list(annotations.keys()))
        random.shuffle(annotated_frame_ids)

        total_frames = len(annotated_frame_ids)
        test_size = int(total_frames * test_ratio)
        val_size = int(total_frames * val_ratio)

        test_ids = set(annotated_frame_ids[:test_size])
        val_ids = set(annotated_frame_ids[test_size : test_size + val_size])
        train_ids = set(annotated_frame_ids[test_size + val_size:])

        # Extract frames and create labels based on the new per-video splits
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            
            if frame_count in annotations:
                img_h, img_w, _ = frame.shape
                frame_name = f"{video_path.stem}_frame_{frame_count:05d}"
                
                # Determine which split this frame belongs to
                if frame_count in train_ids:
                    split = 'train'
                elif frame_count in val_ids:
                    split = 'val'
                elif frame_count in test_ids:
                    split = 'test'
                else:
                    continue # Should not happen

                # Save the frame image
                cv2.imwrite(str(output_path / split / 'images' / f"{frame_name}.jpg"), frame)
                
                yolo_labels = []
                for x1, y1, x2, y2 in annotations[frame_count]:
                    class_id = 0
                    box_w, box_h = x2 - x1, y2 - y1
                    x_center, y_center = x1 + box_w / 2, y1 + box_h / 2
                    yolo_labels.append(f"{class_id} {x_center/img_w} {y_center/img_h} {box_w/img_w} {box_h/img_h}")
                
                with open(output_path / split / 'labels' / f"{frame_name}.txt", 'w') as f:
                    f.write("\n".join(yolo_labels))
        cap.release()

    print("\n--- ✅ YOLO Dataset Preparation Complete! ---")
    print(f"New dataset is ready for training at: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare a single-class YOLO dataset from the F4K dataset.")
    parser.add_argument("--base-path", default="Datasets/f4k", help="Path to the root directory of your source F4K dataset.")
    parser.add_argument("--output-path", default="Datasets/f4k_YOLO", help="Path where the new YOLO-formatted dataset will be created.")
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Ratio of validation data (default: 0.2)')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Ratio of test data (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible splits (default: 42)')
    
    args = parser.parse_args()
    random.seed(args.seed)

    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1.0")

    create_yolo_dataset(args.base_path, args.output_path, args.val_ratio, args.test_ratio)
