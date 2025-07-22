# YOLOv8 Fish Detection Training Pipeline

This pipeline provides configurable training and inference for YOLOv8 fish detection models, supporting both single-class and multi-class modes with minimal configuration changes.

## Features

- **Flexible Training Modes**: Switch between single-class and multi-class detection
- **Single Parameter Control**: Use `single_cls` parameter to control detection mode
- **HPC Ready**: SLURM batch scripts for Orfeo HPC cluster
- **Comprehensive Pipeline**: Training, validation, testing, and inference
- **Easy Configuration**: YAML-based dataset configuration

## Quick Start

### 1. Single-Class Training (Recommended)
```bash
# Train YOLOv8s for single-class fish detection
python train_yolo.py --mode single --model_size s --epochs 100

# Run inference
python inference_yolo.py --model saved_models/yolov8s_single_class_fish.pt --source path/to/images
```

### 2. Multi-Class Training
```bash
# Train YOLOv8s for multi-class fish detection  
python train_yolo.py --mode multi --model_size s --epochs 100

# Run inference
python inference_yolo.py --model saved_models/yolov8s_multi_class_fish.pt --source path/to/images
```

### 3. Orfeo HPC Training
```bash
# Submit training job to Orfeo HPC
sbatch orfeo_train.sh

# Submit inference job
sbatch orfeo_inference.sh
```

## File Structure

```
├── configs/
│   ├── single_class_config.yaml    # Single-class dataset configuration
│   └── multi_class_config.yaml     # Multi-class dataset configuration
├── train_yolo.py                   # Main training script
├── inference_yolo.py               # Inference script
├── orfeo_train.sh                  # SLURM training script for Orfeo HPC
├── orfeo_inference.sh              # SLURM inference script for Orfeo HPC
└── saved_models/                   # Trained model weights
```

## Key Parameters

### The `single_cls` Parameter

The most important parameter for switching between modes:

- **`single_cls=True`**: Treats all classes in multi-class dataset as single class
- **`single_cls=False`**: Uses full multi-class detection

This parameter is automatically set based on the `--mode` argument:
- `--mode single` → `single_cls=True`
- `--mode multi` → `single_cls=False`

## Training Script Usage

```bash
python train_yolo.py [OPTIONS]

Options:
  --mode {single,multi}     Training mode (default: single)
  --model_size {n,s,m,l,x}  YOLO model size (default: s)
  --epochs INT              Number of epochs (default: 100)
  --batch_size INT          Batch size (default: 16)
  --imgsz INT              Image size (default: 640)
  --device STR             Device: auto, cpu, cuda (default: auto)
  --resume STR             Resume from checkpoint
  --pretrained STR         Custom pretrained weights
```

### Examples

```bash
# Basic single-class training
python train_yolo.py --mode single

# Large model with more epochs
python train_yolo.py --mode single --model_size l --epochs 200

# Resume training from checkpoint
python train_yolo.py --mode single --resume runs/train/exp/weights/last.pt

# Multi-class training with custom settings
python train_yolo.py --mode multi --batch_size 32 --epochs 150
```

## Inference Script Usage

```bash
python inference_yolo.py [OPTIONS]

Required:
  --model STR              Path to trained model (.pt file)
  --source STR             Image file, directory, or video

Options:
  --output STR             Output directory (default: runs/detect)
  --conf FLOAT             Confidence threshold (default: 0.25)
  --iou FLOAT              NMS IoU threshold (default: 0.7)
  --max_det INT            Max detections per image (default: 1000)
  --save_txt               Save results as txt files
  --save_conf              Save confidence scores
  --save_crop              Save cropped detections
```

### Examples

```bash
# Single image inference
python inference_yolo.py --model saved_models/best.pt --source image.jpg

# Batch inference on directory
python inference_yolo.py --model saved_models/best.pt --source images/

# Video inference with custom settings
python inference_yolo.py --model saved_models/best.pt --source video.mp4 --conf 0.5

# Save detection crops and annotations
python inference_yolo.py --model saved_models/best.pt --source images/ --save_txt --save_crop
```

## Orfeo HPC Configuration

### Before Submitting Jobs

1. **Update email address** in SLURM scripts:
   ```bash
   #SBATCH --mail-user=your-email@example.com
   ```

2. **Adjust module loading** based on Orfeo's environment:
   ```bash
   module load python/3.9
   module load cuda/11.8
   ```

3. **Configure paths** in the scripts:
   - Model paths
   - Dataset paths  
   - Output directories

### Training on Orfeo

Edit `orfeo_train.sh` configuration variables:
```bash
MODE="single"          # "single" or "multi"
MODEL_SIZE="s"         # "n", "s", "m", "l", "x"  
EPOCHS=100
BATCH_SIZE=16
```

Submit job:
```bash
sbatch orfeo_train.sh
```

Monitor job:
```bash
squeue -u $USER
sacct -j JOB_ID
```

### Inference on Orfeo

Edit `orfeo_inference.sh` configuration:
```bash
MODEL_PATH="saved_models/yolov8s_single_class_fish_orfeo.pt"
SOURCE_PATH="Datasets/Deepfish_YOLO/test/images"
```

Submit job:
```bash
sbatch orfeo_inference.sh
```

## Configuration Files

### Single-Class Config (`configs/single_class_config.yaml`)
```yaml
nc: 1
names: ['Fish']
train: path/to/train/images
val: path/to/val/images  
test: path/to/test/images
```

### Multi-Class Config (`configs/multi_class_config.yaml`)
```yaml
nc: 23
names: ['Caranx_sexfasciatus', 'F1', 'F2', ...]
train: path/to/train/images
val: path/to/val/images
test: path/to/test/images
```

## Model Performance Optimization

### Single-Class Mode Optimizations
- `single_cls=True`: Treats all detections as single class
- `cls=0.5`: Lower classification loss weight
- `box=7.5`: Higher box regression focus
- `mixup=0.0`: No mixup augmentation

### Multi-Class Mode Optimizations  
- `single_cls=False`: Full multi-class detection
- `cls=1.0`: Higher classification loss weight
- `mixup=0.15`: Higher mixup for better generalization
- `copy_paste=0.3`: Increased copy-paste augmentation

## Output Structure

```
runs/train/experiment_name/
├── weights/
│   ├── best.pt          # Best model weights
│   ├── last.pt          # Last epoch weights
│   └── epoch_*.pt       # Periodic checkpoints
├── results.png          # Training curves
├── confusion_matrix.png # Confusion matrix
├── val_batch*.jpg       # Validation predictions
└── args.yaml           # Training arguments

saved_models/
└── yolov8s_single_class_fish.pt  # Final model for deployment
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `--batch_size`
   - Use smaller model (`--model_size n`)
   - Reduce `--imgsz`

2. **Dataset Path Errors**:
   - Check paths in config YAML files
   - Ensure train/val/test directories exist
   - Verify image and label file pairing

3. **Model Loading Errors**:
   - Check model file exists
   - Verify file permissions
   - Ensure compatible PyTorch/Ultralytics versions

### Performance Tips

1. **For Better Speed**:
   - Use smaller models (YOLOv8n)
   - Reduce image size (`--imgsz 416`)
   - Increase batch size if memory allows

2. **For Better Accuracy**:
   - Use larger models (YOLOv8l, YOLOv8x)
   - Increase epochs (`--epochs 200+`)
   - Use data augmentation
   - Fine-tune confidence thresholds

## Dependencies

```bash
pip install ultralytics torch torchvision opencv-python pillow pyyaml matplotlib
```

## License

This project follows the same license as the original YOLOv8 implementation.