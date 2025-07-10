#!/usr/bin/env python3
"""
Train DeepFish on Mac M3 - Final Version
Optimized training script for DeepFish YOLO dataset on Apple Silicon M3.
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import time
import json
import matplotlib.pyplot as plt
import psutil
from datetime import datetime
import yaml

class DeepFishMacTrainer:
    """Train DeepFish model optimized for Mac M3."""
    
    def __init__(self, dataset_dir="deepfish_yolo"):
        self.dataset_dir = Path(dataset_dir)
        self.results_dir = Path("training_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Check Mac M3 setup
        self.check_mac_m3_setup()
        
    def check_mac_m3_setup(self):
        """Check Mac M3 capabilities for training."""
        print("🍎 MAC M3 TRAINING SETUP")
        print("=" * 30)
        
        # Check PyTorch MPS support
        if torch.backends.mps.is_available():
            print("✅ Metal Performance Shaders (MPS) available")
            self.device = 'mps'
        else:
            print("❌ MPS not available - using CPU")
            print("💡 Install PyTorch with MPS: pip install torch torchvision torchaudio")
            self.device = 'cpu'
        
        # Memory check
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"💾 Total RAM: {memory_gb:.1f} GB")
        
        # Set optimal batch size based on memory
        if memory_gb >= 32:
            self.batch_size = 32
            print("🚀 High memory detected - using large batch size")
        elif memory_gb >= 16:
            self.batch_size = 16
            print("✅ Good memory for training")
        else:
            self.batch_size = 8
            print("⚠️  Limited memory - using smaller batch size")
        
        print(f"🎯 Optimal batch size: {self.batch_size}")
        
        # CPU cores for data loading
        cpu_cores = psutil.cpu_count()
        self.workers = min(cpu_cores // 2, 8)  # Use half the cores, max 8
        print(f"🔄 Data loading workers: {self.workers}")
        
        # Test MPS functionality
        if self.device == 'mps':
            try:
                test_tensor = torch.randn(100, 100).to('mps')
                result = torch.matmul(test_tensor, test_tensor)
                print("✅ MPS functionality test passed")
            except Exception as e:
                print(f"❌ MPS test failed: {e}")
                print("Falling back to CPU")
                self.device = 'cpu'
        
        # Free up some memory
        if self.device == 'mps':
            torch.mps.empty_cache()
    
    def validate_dataset(self):
        """Validate the converted DeepFish YOLO dataset."""
        print(f"\n🔍 VALIDATING YOLO DATASET")
        print("=" * 30)
        
        # Check main directory
        if not self.dataset_dir.exists():
            print(f"❌ Dataset directory not found: {self.dataset_dir}")
            print("💡 Make sure you ran the DeepFish converter first")
            return False
        
        # Check data.yaml
        config_file = self.dataset_dir / "data.yaml"
        if not config_file.exists():
            print(f"❌ Config file not found: {config_file}")
            return False
        
        print(f"✅ Found config file: {config_file}")
        
        # Load and validate config
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"📋 Dataset configuration:")
            print(f"   Classes: {config.get('nc', 'unknown')}")
            print(f"   Names: {config.get('names', 'unknown')}")
            
        except Exception as e:
            print(f"⚠️  Error reading config: {e}")
        
        # Check splits and count images
        splits_info = {}
        total_images = 0
        
        for split in ['train', 'val', 'test']:
            img_dir = self.dataset_dir / 'images' / split
            label_dir = self.dataset_dir / 'labels' / split
            
            if img_dir.exists() and label_dir.exists():
                img_count = len(list(img_dir.glob('*.jpg')))
                label_count = len(list(label_dir.glob('*.txt')))
                
                splits_info[split] = {
                    'images': img_count,
                    'labels': label_count,
                    'matched': img_count == label_count
                }
                
                total_images += img_count
                
                status = "✅" if img_count == label_count else "⚠️"
                print(f"{status} {split}: {img_count} images, {label_count} labels")
                
                if img_count != label_count:
                    print(f"   ⚠️  Mismatch between images and labels!")
            else:
                print(f"❌ {split}: Missing directories")
                return False
        
        print(f"📊 Total dataset size: {total_images} images")
        
        if total_images < 100:
            print("⚠️  Very small dataset - training may not be effective")
            print("💡 Consider using more data or reducing epochs")
        elif total_images < 1000:
            print("⚠️  Small dataset - will use more data augmentation")
        else:
            print("✅ Good dataset size for training")
        
        self.dataset_size = total_images
        return True
    
    def setup_training_config(self):
        """Setup training configuration optimized for Mac M3 and dataset size."""
        print(f"\n⚙️  TRAINING CONFIGURATION")
        print("=" * 30)
        
        # Adjust epochs based on dataset size
        if self.dataset_size < 500:
            self.epochs = 200  # More epochs for small dataset
            print("📈 Small dataset - using more epochs")
        elif self.dataset_size < 2000:
            self.epochs = 100  # Standard epochs
            print("📈 Medium dataset - using standard epochs")
        else:
            self.epochs = 80   # Fewer epochs for large dataset
            print("📈 Large dataset - using fewer epochs")
        
        # Other M3-optimized settings
        self.imgsz = 640       # Good balance for M3
        self.patience = 30     # Generous early stopping for small datasets
        self.save_period = 20  # Save checkpoint every 20 epochs
        
        # Learning rate - start higher for small datasets
        if self.dataset_size < 1000:
            self.lr0 = 0.01    # Higher learning rate
        else:
            self.lr0 = 0.01    # Standard learning rate
        
        config = {
            'device': self.device,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'imgsz': self.imgsz,
            'workers': self.workers,
            'patience': self.patience,
            'save_period': self.save_period,
            'lr0': self.lr0,
            'dataset_size': self.dataset_size
        }
        
        print(f"🎯 Training configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        return config
    
    def train_model(self, model_size='n', resume=False):
        """Train the fish detection model."""
        print(f"\n🚀 STARTING DEEPFISH TRAINING")
        print("=" * 35)
        
        # Validate dataset first
        if not self.validate_dataset():
            print("❌ Dataset validation failed")
            return None
        
        # Setup configuration
        config = self.setup_training_config()
        
        # Load pre-trained model
        model_name = f'yolov8{model_size}.pt'
        print(f"📥 Loading pre-trained model: {model_name}")
        
        try:
            model = YOLO(model_name)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
        
        # Create unique experiment name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"deepfish_mac_m3_{model_size}_{timestamp}"
        
        print(f"🏋️  Starting training...")
        print(f"📁 Results will be saved to: {self.results_dir / experiment_name}")
        print(f"⏱️  Estimated training time: {self.epochs * self.dataset_size / (self.batch_size * 3600):.1f} hours")
        
        start_time = time.time()
        
        try:
            # Train the model with Mac M3 optimizations
            results = model.train(
                data=str(self.dataset_dir / "data.yaml"),
                epochs=self.epochs,
                imgsz=self.imgsz,
                batch=self.batch_size,
                device=self.device,
                workers=self.workers,
                patience=self.patience,
                save=True,
                save_period=self.save_period,
                plots=True,
                val=True,
                project=str(self.results_dir),
                name=experiment_name,
                resume=resume,
                
                # Learning rate settings
                lr0=self.lr0,
                lrf=0.01,  # Final learning rate
                
                # Mac M3 specific optimizations
                amp=False,  # Disable AMP for MPS stability
                verbose=True,
                
                # Data augmentation (enhanced for underwater scenes)
                hsv_h=0.015,    # Hue augmentation (underwater color shifts)
                hsv_s=0.7,      # Saturation augmentation  
                hsv_v=0.4,      # Value augmentation (lighting changes)
                degrees=15.0,   # Rotation augmentation
                translate=0.1,  # Translation augmentation
                scale=0.5,      # Scale augmentation
                shear=0.0,      # No shear (fish shapes are important)
                perspective=0.0, # No perspective (underwater distortion)
                flipud=0.0,     # No vertical flip (fish orientation matters)
                fliplr=0.5,     # Horizontal flip (fish can face either direction)
                mosaic=1.0,     # Mosaic augmentation
                mixup=0.1,      # Mixup augmentation
                copy_paste=0.0, # No copy-paste for underwater scenes
                
                # Optimizer settings for small datasets
                optimizer='AdamW',  # Better for small datasets
                weight_decay=0.0005,
                
                # Loss function weights
                cls=0.5,        # Classification loss weight
                box=7.5,        # Box regression loss weight
                dfl=1.5,        # Distribution focal loss weight
            )
            
            training_time = time.time() - start_time
            
            print(f"\n🎉 TRAINING COMPLETED!")
            print(f"⏱️  Total training time: {training_time/3600:.2f} hours")
            print(f"📁 Results saved to: {self.results_dir / experiment_name}")
            
            # Find the best model
            best_model_path = self.results_dir / experiment_name / "weights" / "best.pt"
            if best_model_path.exists():
                print(f"🏆 Best model saved: {best_model_path}")
            
            # Save training summary
            self.save_training_summary(experiment_name, config, training_time, results)
            
            # Run quick evaluation
            self.quick_evaluation(best_model_path if best_model_path.exists() else None)
            
            return results
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            print(f"\n💡 Troubleshooting tips:")
            print(f"   - Try smaller batch size: --batch 8")
            print(f"   - Try smaller image size: --imgsz 416")
            print(f"   - Check dataset format")
            print(f"   - Restart Python to clear memory")
            return None
    
    def save_training_summary(self, experiment_name, config, training_time, results):
        """Save a summary of the training session."""
        summary = {
            'experiment_name': experiment_name,
            'training_date': datetime.now().isoformat(),
            'device': config['device'],
            'training_time_hours': training_time / 3600,
            'config': config,
            'mac_specs': {
                'total_ram_gb': psutil.virtual_memory().total / (1024**3),
                'cpu_cores': psutil.cpu_count(),
                'mps_available': torch.backends.mps.is_available()
            }
        }
        
        # Add results metrics if available
        if results:
            try:
                # Look for results CSV
                results_dir = self.results_dir / experiment_name
                results_csv = results_dir / "results.csv"
                
                if results_csv.exists():
                    import pandas as pd
                    df = pd.read_csv(results_csv)
                    if len(df) > 0:
                        last_row = df.iloc[-1]
                        summary['final_metrics'] = {
                            'mAP50': float(last_row.get('metrics/mAP50(B)', 0)),
                            'mAP50-95': float(last_row.get('metrics/mAP50-95(B)', 0)),
                            'precision': float(last_row.get('metrics/precision(B)', 0)),
                            'recall': float(last_row.get('metrics/recall(B)', 0)),
                            'final_epoch': int(last_row.get('epoch', 0))
                        }
                        
                        print(f"\n📊 FINAL METRICS:")
                        print(f"   mAP@0.5: {summary['final_metrics']['mAP50']:.3f}")
                        print(f"   mAP@0.5:0.95: {summary['final_metrics']['mAP50-95']:.3f}")
                        print(f"   Precision: {summary['final_metrics']['precision']:.3f}")
                        print(f"   Recall: {summary['final_metrics']['recall']:.3f}")
                        
            except Exception as e:
                print(f"Could not extract final metrics: {e}")
        
        # Save summary
        summary_path = self.results_dir / f"{experiment_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Training summary saved: {summary_path}")
    
    def quick_evaluation(self, model_path):
        """Run quick evaluation on the trained model."""
        if not model_path or not Path(model_path).exists():
            print("⚠️  No model available for evaluation")
            return
        
        print(f"\n🧪 QUICK EVALUATION")
        print("=" * 25)
        
        try:
            # Load trained model
            model = YOLO(str(model_path))
            
            # Run validation
            val_results = model.val(
                data=str(self.dataset_dir / "data.yaml"),
                device=self.device,
                verbose=False
            )
            
            print(f"✅ Validation completed")
            
            # Test on a few random images from validation set
            val_img_dir = self.dataset_dir / 'images' / 'val'
            if val_img_dir.exists():
                test_images = list(val_img_dir.glob('*.jpg'))[:5]  # Test 5 images
                
                if test_images:
                    print(f"🎯 Testing on {len(test_images)} validation images...")
                    
                    for img_path in test_images:
                        results = model(str(img_path), conf=0.25, verbose=False)
                        detections = len(results[0].boxes) if results[0].boxes is not None else 0
                        print(f"   {img_path.name}: {detections} fish detected")
        
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
    
    def resume_training(self, experiment_name):
        """Resume interrupted training."""
        print(f"🔄 RESUMING TRAINING: {experiment_name}")
        
        resume_path = self.results_dir / experiment_name / "weights" / "last.pt"
        
        if not resume_path.exists():
            print(f"❌ Resume checkpoint not found: {resume_path}")
            return None
        
        print(f"📥 Resuming from: {resume_path}")
        
        try:
            model = YOLO(str(resume_path))
            results = model.train(resume=True)
            return results
        except Exception as e:
            print(f"❌ Resume failed: {e}")
            return None

def main():
    """Main training function."""
    print("🐟 DEEPFISH MAC M3 TRAINER")
    print("=" * 30)
    
    # Get dataset directory
    dataset_dir = input("Enter YOLO dataset directory (or press Enter for 'deepfish_yolo'): ").strip()
    if not dataset_dir:
        dataset_dir = "deepfish_yolo"
    
    # Get model size
    print(f"\nModel size options:")
    print(f"  n = nano (6MB, fastest)")
    print(f"  s = small (22MB, good balance)")
    print(f"  m = medium (52MB, more accurate)")
    
    model_size = input("Choose model size (n/s/m, default 'n'): ").strip().lower()
    if model_size not in ['n', 's', 'm']:
        model_size = 'n'
    
    # Initialize trainer
    trainer = DeepFishMacTrainer(dataset_dir)
    
    # Check if resuming
    resume = input("Resume previous training? (y/n, default 'n'): ").strip().lower() == 'y'
    
    if resume:
        # List available experiments
        experiments = list(trainer.results_dir.glob("deepfish_mac_m3_*"))
        if experiments:
            print(f"\nAvailable experiments:")
            for i, exp in enumerate(experiments):
                print(f"  {i+1}. {exp.name}")
            
            choice = input("Select experiment number: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(experiments):
                exp_name = experiments[int(choice)-1].name
                results = trainer.resume_training(exp_name)
            else:
                print("Invalid choice")
                return
        else:
            print("No previous experiments found")
            return
    else:
        # Start new training
        results = trainer.train_model(model_size=model_size)
    
    if results:
        print(f"\n✅ Training completed successfully!")
        print(f"\n🚀 Next steps:")
        print(f"1. Check results in: {trainer.results_dir}")
        print(f"2. Test model on your videos")
        print(f"3. Integrate with ByteTrack for tracking")
    else:
        print(f"\n❌ Training failed")

if __name__ == "__main__":
    main()
