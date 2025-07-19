#!/bin/bash
#SBATCH --job-name=yolo_fish_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/yolo_train_%j.out
#SBATCH --error=logs/yolo_train_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your-email@example.com

# Configuration variables - MODIFY THESE AS NEEDED
MODE="single"                    # "single" or "multi" 
MODEL_SIZE="s"                   # "n", "s", "m", "l", "x"
EPOCHS=100
BATCH_SIZE=16
IMGSZ=640
PROJECT_NAME="deepfish_orfeo"

# Create necessary directories
mkdir -p logs
mkdir -p saved_models

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

# Install/upgrade required packages if needed
echo "Installing/upgrading required packages..."
pip install --upgrade ultralytics torch torchvision torchaudio
pip install --upgrade opencv-python pillow pyyaml matplotlib

# Run training
echo "Starting YOLOv8 training..."
echo "Mode: $MODE"
echo "Model size: $MODEL_SIZE"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Image size: $IMGSZ"
echo "=================================================="

# Execute training script
python train_yolo.py \
    --mode "$MODE" \
    --model_size "$MODEL_SIZE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --imgsz "$IMGSZ" \
    --device cuda \
    --project runs/train \
    --name "${PROJECT_NAME}_${MODE}_${MODEL_SIZE}"

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "=================================================="
    echo "Training completed successfully!"
    echo "End time: $(date)"
    
    # Copy best model to saved_models with descriptive name
    EXPERIMENT_NAME="${PROJECT_NAME}_${MODE}_${MODEL_SIZE}"
    BEST_MODEL="runs/train/${EXPERIMENT_NAME}/weights/best.pt"
    OUTPUT_MODEL="saved_models/yolov8${MODEL_SIZE}_${MODE}_class_fish_orfeo.pt"
    
    if [ -f "$BEST_MODEL" ]; then
        cp "$BEST_MODEL" "$OUTPUT_MODEL"
        echo "Best model copied to: $OUTPUT_MODEL"
    else
        echo "Warning: Best model not found at $BEST_MODEL"
    fi
    
    # Print final model statistics
    echo "=================================================="
    echo "Model Training Summary:"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Mode: $MODE-class"
    echo "Model: YOLOv8$MODEL_SIZE"
    echo "Epochs: $EPOCHS"
    echo "Batch size: $BATCH_SIZE"
    echo "Final model: $OUTPUT_MODEL"
    echo "Training directory: runs/train/${EXPERIMENT_NAME}"
    
    # Run inference test on a sample image if available
    TEST_IMAGE="Datasets/Deepfish_YOLO/test_image.jpg"
    if [ -f "$TEST_IMAGE" ]; then
        echo "=================================================="
        echo "Running inference test on sample image..."
        python inference_yolo.py \
            --model "$OUTPUT_MODEL" \
            --source "$TEST_IMAGE" \
            --output "runs/inference_test" \
            --name "orfeo_test_${MODE}_${MODEL_SIZE}" \
            --conf 0.25 \
            --device cuda
    fi
    
else
    echo "=================================================="
    echo "Training failed with exit code: $?"
    echo "End time: $(date)"
    echo "Check the error log for details."
    exit 1
fi

echo "=================================================="
echo "Job completed at: $(date)"
echo "=================================================="