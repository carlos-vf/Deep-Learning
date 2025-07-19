#!/bin/bash
#SBATCH --job-name=yolo_fish_inference
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/yolo_inference_%j.out
#SBATCH --error=logs/yolo_inference_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your-email@example.com

# Configuration variables - MODIFY THESE AS NEEDED
MODEL_PATH="saved_models/yolov8s_single_class_fish_orfeo.pt"    # Path to trained model
SOURCE_PATH="Datasets/Deepfish_YOLO/test/images"               # Input: image file, directory, or video
OUTPUT_DIR="runs/inference"                                     # Output directory
CONF_THRESHOLD=0.25                                            # Confidence threshold
IOU_THRESHOLD=0.7                                              # NMS IoU threshold
MAX_DETECTIONS=1000                                            # Maximum detections per image
EXPERIMENT_NAME="orfeo_inference"

# Create necessary directories
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# Print job information
echo "=================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo "=================================================="

# Load required modules (adjust according to Orfeo's module system)
# Uncomment and modify these lines based on Orfeo's available modules
# module load python/3.9
# module load cuda/11.8
# module load cudnn/8.6

# Activate virtual environment if using one
# source /path/to/your/venv/bin/activate

# Print Python and CUDA information
echo "Python version:"
python --version
echo "PyTorch CUDA available:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
echo "=================================================="

# Navigate to project directory
cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    echo "Available models in saved_models/:"
    ls -la saved_models/ 2>/dev/null || echo "No saved_models directory found"
    exit 1
fi

# Check if source exists
if [ ! -e "$SOURCE_PATH" ]; then
    echo "Error: Source path not found at $SOURCE_PATH"
    exit 1
fi

# Run inference
echo "Starting YOLOv8 inference..."
echo "Model: $MODEL_PATH"
echo "Source: $SOURCE_PATH"
echo "Output: $OUTPUT_DIR"
echo "Confidence threshold: $CONF_THRESHOLD"
echo "IoU threshold: $IOU_THRESHOLD"
echo "Max detections: $MAX_DETECTIONS"
echo "=================================================="

# Execute inference script
python inference_yolo.py \
    --model "$MODEL_PATH" \
    --source "$SOURCE_PATH" \
    --output "$OUTPUT_DIR" \
    --name "$EXPERIMENT_NAME" \
    --conf "$CONF_THRESHOLD" \
    --iou "$IOU_THRESHOLD" \
    --max_det "$MAX_DETECTIONS" \
    --device cuda \
    --save_txt \
    --save_conf \
    --show_labels \
    --show_conf

# Check if inference completed successfully
if [ $? -eq 0 ]; then
    echo "=================================================="
    echo "Inference completed successfully!"
    echo "End time: $(date)"
    
    # Print results summary
    RESULTS_DIR="$OUTPUT_DIR/$EXPERIMENT_NAME"
    echo "=================================================="
    echo "Inference Results Summary:"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Model: $MODEL_PATH"
    echo "Source: $SOURCE_PATH"
    echo "Results directory: $RESULTS_DIR"
    
    # Count processed files
    if [ -d "$RESULTS_DIR" ]; then
        IMAGE_COUNT=$(find "$RESULTS_DIR" -name "*.jpg" -o -name "*.png" | wc -l)
        TXT_COUNT=$(find "$RESULTS_DIR" -name "*.txt" | wc -l)
        echo "Images processed: $IMAGE_COUNT"
        echo "Detection files created: $TXT_COUNT"
        
        # Show directory contents
        echo "Output directory contents:"
        ls -la "$RESULTS_DIR" | head -20
        
        if [ $IMAGE_COUNT -gt 20 ]; then
            echo "... and $(($IMAGE_COUNT - 20)) more files"
        fi
    else
        echo "Warning: Results directory not found"
    fi
    
else
    echo "=================================================="
    echo "Inference failed with exit code: $?"
    echo "End time: $(date)"
    echo "Check the error log for details."
    exit 1
fi

echo "=================================================="
echo "Job completed at: $(date)"
echo "=================================================="