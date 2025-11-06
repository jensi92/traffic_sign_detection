"""
GTSRB Traffic Sign Detection - Training Module

This module implements the training pipeline for YOLOv8 model:
- Model initialization and configuration
- Training loop with callbacks
- Learning rate scheduling
- Model checkpointing
- Training visualization and logging

YOLOv8 Architecture Overview:
- Backbone: CSPDarknet for feature extraction
- Neck: PANet (Path Aggregation Network) for multi-scale features
- Head: Decoupled detection head for classification and localization
"""

import os
import torch
from pathlib import Path
from ultralytics import YOLO
import time
from typing import Dict, Any

from utils import Logger, load_config, load_class_names, format_time


class TrafficSignTrainer:
    """Trainer class for traffic sign detection using YOLOv8."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = Logger(
            self.config['output']['logs_dir'],
            name="training"
        )
        self.class_names = load_class_names()
        
        # Setup paths
        self.dataset_yaml = Path(self.config['data']['output_dir']) / 'dataset.yaml'
        self.weights_dir = Path(self.config['output']['weights_dir'])
        self.runs_dir = Path(self.config['output']['runs_dir'])
        
        # Create directories
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if dataset is prepared
        if not self.dataset_yaml.exists():
            raise FileNotFoundError(
                f"Dataset YAML not found at {self.dataset_yaml}. "
                "Please run data_preprocessing.py first."
            )
        
        self.logger.info(f"Initialized trainer with {len(self.class_names)} classes")
        self.logger.info(f"Dataset YAML: {self.dataset_yaml}")
    
    def initialize_model(self) -> YOLO:
        """
        Initialize YOLOv8 model.
        
        YOLOv8 is a state-of-the-art object detection model that uses:
        - Anchor-free detection (no predefined anchor boxes)
        - Mosaic augmentation for better generalization
        - CIoU loss for better bounding box regression
        
        Returns:
            YOLO model instance
        """
        model_arch = self.config['model']['architecture']
        self.logger.info(f"Initializing YOLOv8 model: {model_arch}")
        
        try:
            # Load pretrained YOLOv8 model
            # The model will be downloaded automatically if not present
            model = YOLO(model_arch)
            self.logger.info("Model initialized successfully")
            
            # Log model info
            self.logger.info(f"Model parameters: {sum(p.numel() for p in model.model.parameters()):,}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise
    
    def train(self) -> Dict[str, Any]:
        """
        Train the YOLOv8 model.
        
        Training Process:
        1. Data Loading: Images are loaded and augmented on-the-fly
        2. Forward Pass: Model predicts bounding boxes and class probabilities
        3. Loss Calculation: CIoU loss (box) + BCE loss (classification)
        4. Backward Pass: Gradients computed and weights updated
        5. Validation: Model evaluated on validation set periodically
        
        Returns:
            Dictionary with training results
        """
        self.logger.info("="*80)
        self.logger.info("STARTING TRAINING")
        self.logger.info("="*80)
        
        # Initialize model
        model = self.initialize_model()
        
        # Get training parameters
        epochs = self.config['training']['epochs']
        batch_size = self.config['training']['batch_size']
        img_size = self.config['model']['img_size']
        lr = self.config['training']['learning_rate']
        device = self.config['training']['device']
        patience = self.config['training']['patience']
        workers = self.config['training']['workers']
        
        # Log training configuration
        self.logger.info("\nTraining Configuration:")
        self.logger.info(f"  Epochs: {epochs}")
        self.logger.info(f"  Batch Size: {batch_size}")
        self.logger.info(f"  Image Size: {img_size}")
        self.logger.info(f"  Learning Rate: {lr}")
        self.logger.info(f"  Device: {device}")
        self.logger.info(f"  Workers: {workers}")
        self.logger.info(f"  Patience (Early Stopping): {patience}")
        
        # Start training timer
        start_time = time.time()
        
        try:
            # Train the model
            # YOLOv8's train() method handles:
            # - Data loading and augmentation
            # - Training loop with automatic mixed precision (AMP)
            # - Learning rate scheduling (cosine annealing)
            # - Model checkpointing
            # - Validation and metric computation
            # - TensorBoard logging
            
            results = model.train(
                data=str(self.dataset_yaml),
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                lr0=lr,
                device=device,
                patience=patience,
                workers=workers,
                cache=False,  # Disable caching to avoid RAM issues
                project=str(self.runs_dir),
                name='train',
                exist_ok=True,
                pretrained=True,
                optimizer=self.config['training']['optimizer'],
                verbose=True,
                seed=self.config['validation']['random_seed'],
                deterministic=True,
                single_cls=False,
                rect=False,
                cos_lr=True,
                close_mosaic=10,
                amp=True,  # Automatic Mixed Precision for faster training
                # Data augmentation parameters
                hsv_h=self.config['augmentation']['hsv_h'],
                hsv_s=self.config['augmentation']['hsv_s'],
                hsv_v=self.config['augmentation']['hsv_v'],
                degrees=self.config['augmentation']['degrees'],
                translate=self.config['augmentation']['translate'],
                scale=self.config['augmentation']['scale'],
                shear=self.config['augmentation']['shear'],
                perspective=self.config['augmentation']['perspective'],
                flipud=self.config['augmentation']['flipud'],
                fliplr=self.config['augmentation']['fliplr'],
                mosaic=self.config['augmentation']['mosaic'],
                mixup=self.config['augmentation']['mixup'],
                # Validation parameters
                val=True,
                save=True,
                save_period=self.config['training']['save_period'],
                plots=True,
                # Loss weights (default values work well)
                box=7.5,
                cls=0.5,
                dfl=1.5,
            )
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Log training completion
            self.logger.info("\n" + "="*80)
            self.logger.info("TRAINING COMPLETED")
            self.logger.info("="*80)
            self.logger.info(f"Total training time: {format_time(training_time)}")
            self.logger.info(f"Best model saved at: {model.trainer.best}")
            self.logger.info(f"Last model saved at: {model.trainer.last}")
            
            # Log final metrics
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                self.logger.info("\nFinal Metrics:")
                self.logger.info(f"  mAP@0.5: {metrics.get('metrics/mAP50(B)', 0):.4f}")
                self.logger.info(f"  mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
                self.logger.info(f"  Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
                self.logger.info(f"  Recall: {metrics.get('metrics/recall(B)', 0):.4f}")
            
            # Copy best weights to weights directory
            best_weights = Path(model.trainer.best)
            if best_weights.exists():
                import shutil
                dest_path = self.weights_dir / 'best.pt'
                shutil.copy2(best_weights, dest_path)
                self.logger.info(f"\n✓ Best weights copied to: {dest_path}")
            
            return {
                'success': True,
                'training_time': training_time,
                'best_weights': str(model.trainer.best),
                'last_weights': str(model.trainer.last),
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise
    
    def resume_training(self, weights_path: str) -> Dict[str, Any]:
        """
        Resume training from a checkpoint.
        
        Args:
            weights_path: Path to checkpoint weights
            
        Returns:
            Dictionary with training results
        """
        self.logger.info(f"Resuming training from: {weights_path}")
        
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        # Load model from checkpoint
        model = YOLO(weights_path)
        
        # Continue training with the same configuration
        # The model will resume from the checkpoint's epoch
        results = model.train(
            data=str(self.dataset_yaml),
            resume=True,
            project=str(self.runs_dir),
            name='train',
            exist_ok=True
        )
        
        return {
            'success': True,
            'results': results
        }


def main():
    """Main function to run training."""
    try:
        # Initialize trainer
        trainer = TrafficSignTrainer()
        
        # Train model
        results = trainer.train()
        
        if results['success']:
            print("\n" + "="*80)
            print("✓ TRAINING COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"Training Time: {format_time(results['training_time'])}")
            print(f"Best Weights: {results['best_weights']}")
            print(f"Last Weights: {results['last_weights']}")
            print(f"\nYou can now:")
            print("  1. Run evaluate_model.py to evaluate the model")
            print("  2. Run predict.py to make predictions on new images")
            print("  3. Run app.py to launch the web interface")
            print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
