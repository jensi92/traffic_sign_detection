"""
GTSRB Traffic Sign Detection - Evaluation Module

This module provides comprehensive model evaluation:
- Mean Average Precision (mAP) at different IoU thresholds
- Precision and Recall curves
- Confusion matrix
- Per-class performance metrics
- Speed/throughput benchmarking
"""

import os
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

from utils import (
    Logger, load_config, load_class_names,
    plot_confusion_matrix, format_time
)


class TrafficSignEvaluator:
    """Evaluator class for traffic sign detection model."""
    
    def __init__(
        self,
        weights_path: str,
        config_path: str = "config.yaml"
    ):
        """
        Initialize the evaluator.
        
        Args:
            weights_path: Path to trained model weights
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = Logger(
            self.config['output']['logs_dir'],
            name="evaluation"
        )
        self.class_names = load_class_names()
        
        # Setup paths
        self.dataset_yaml = Path(self.config['data']['output_dir']) / 'dataset.yaml'
        self.predictions_dir = Path(self.config['output']['predictions_dir'])
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        
        # Check weights file
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        # Load model
        self.logger.info(f"Loading model from: {weights_path}")
        self.model = YOLO(weights_path)
        self.logger.info("Model loaded successfully")
    
    def validate(self) -> Dict[str, Any]:
        """
        Run validation on the validation set.
        
        This computes standard object detection metrics:
        - mAP@0.5: Mean Average Precision at IoU threshold 0.5
        - mAP@0.5:0.95: Mean Average Precision averaged over IoU thresholds from 0.5 to 0.95
        - Precision: TP / (TP + FP)
        - Recall: TP / (TP + FN)
        
        Returns:
            Dictionary with validation metrics
        """
        self.logger.info("="*80)
        self.logger.info("RUNNING VALIDATION")
        self.logger.info("="*80)
        
        try:
            # Run validation
            # YOLOv8's val() method computes:
            # - Box loss: CIoU loss for bounding box regression
            # - Class loss: Binary cross-entropy for classification
            # - DFL loss: Distribution Focal Loss for box distribution
            
            results = self.model.val(
                data=str(self.dataset_yaml),
                imgsz=self.config['model']['img_size'],
                batch=self.config['training']['batch_size'],
                conf=self.config['evaluation']['conf_threshold'],
                iou=self.config['evaluation']['iou_threshold'],
                max_det=self.config['evaluation']['max_det'],
                device=self.config['training']['device'],
                workers=self.config['training']['workers'],
                plots=True,
                save_json=True,
                save_hybrid=False,
                verbose=True
            )
            
            # Extract metrics
            metrics = {
                'mAP50': results.box.map50,  # mAP@0.5
                'mAP50-95': results.box.map,  # mAP@0.5:0.95
                'precision': results.box.mp,   # Mean precision
                'recall': results.box.mr,      # Mean recall
                'box_loss': results.box.box_loss if hasattr(results.box, 'box_loss') else None,
                'cls_loss': results.box.cls_loss if hasattr(results.box, 'cls_loss') else None,
                'dfl_loss': results.box.dfl_loss if hasattr(results.box, 'dfl_loss') else None,
            }
            
            # Log metrics
            self.logger.info("\n" + "="*80)
            self.logger.info("VALIDATION METRICS")
            self.logger.info("="*80)
            self.logger.info(f"mAP@0.5: {metrics['mAP50']:.4f}")
            self.logger.info(f"mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
            self.logger.info(f"Precision: {metrics['precision']:.4f}")
            self.logger.info(f"Recall: {metrics['recall']:.4f}")
            
            if metrics['box_loss'] is not None:
                self.logger.info(f"\nLosses:")
                self.logger.info(f"  Box Loss: {metrics['box_loss']:.4f}")
                self.logger.info(f"  Class Loss: {metrics['cls_loss']:.4f}")
                self.logger.info(f"  DFL Loss: {metrics['dfl_loss']:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during validation: {e}")
            raise
    
    def evaluate_per_class(self) -> pd.DataFrame:
        """
        Evaluate performance for each class.
        
        Returns:
            DataFrame with per-class metrics
        """
        self.logger.info("\nComputing per-class metrics...")
        
        # Run validation to get detailed results
        results = self.model.val(
            data=str(self.dataset_yaml),
            imgsz=self.config['model']['img_size'],
            conf=self.config['evaluation']['conf_threshold'],
            iou=self.config['evaluation']['iou_threshold'],
            device=self.config['training']['device'],
            verbose=False
        )
        
        # Extract per-class metrics
        class_metrics = []
        
        # Get per-class AP if available
        if hasattr(results.box, 'ap_class_index'):
            ap_per_class = results.box.ap  # AP for each class
            ap_class_index = results.box.ap_class_index  # Class indices
            
            for idx, class_id in enumerate(ap_class_index):
                class_metrics.append({
                    'Class ID': int(class_id),
                    'Class Name': self.class_names[int(class_id)],
                    'AP@0.5': float(ap_per_class[idx, 0]) if len(ap_per_class.shape) > 1 else float(ap_per_class[idx]),
                    'AP@0.5:0.95': float(results.box.map_per_class[idx]) if hasattr(results.box, 'map_per_class') else 0.0
                })
        
        df = pd.DataFrame(class_metrics)
        
        if len(df) > 0:
            # Sort by AP@0.5 descending
            df = df.sort_values('AP@0.5', ascending=False)
            
            # Log top and bottom performers
            self.logger.info("\nTop 10 Best Performing Classes:")
            for idx, row in df.head(10).iterrows():
                self.logger.info(f"  {row['Class Name']}: AP@0.5 = {row['AP@0.5']:.4f}")
            
            self.logger.info("\nTop 10 Worst Performing Classes:")
            for idx, row in df.tail(10).iterrows():
                self.logger.info(f"  {row['Class Name']}: AP@0.5 = {row['AP@0.5']:.4f}")
            
            # Save to CSV
            csv_path = self.predictions_dir / 'per_class_metrics.csv'
            df.to_csv(csv_path, index=False)
            self.logger.info(f"\nPer-class metrics saved to: {csv_path}")
        
        return df
    
    def benchmark_speed(self, num_images: int = 100) -> Dict[str, float]:
        """
        Benchmark model inference speed.
        
        Args:
            num_images: Number of images to test
            
        Returns:
            Dictionary with speed metrics
        """
        self.logger.info("\n" + "="*80)
        self.logger.info(f"BENCHMARKING SPEED (on {num_images} images)")
        self.logger.info("="*80)
        
        # Get validation image paths
        val_images_dir = Path(self.config['data']['output_dir']) / 'images' / 'val'
        image_paths = list(val_images_dir.glob('*.png'))[:num_images]
        
        if len(image_paths) == 0:
            self.logger.warning("No validation images found for benchmarking")
            return {}
        
        # Warm-up run
        self.logger.info("Warming up...")
        for img_path in image_paths[:10]:
            self.model.predict(
                str(img_path),
                conf=self.config['evaluation']['conf_threshold'],
                verbose=False
            )
        
        # Benchmark
        self.logger.info(f"Running inference on {len(image_paths)} images...")
        start_time = time.time()
        
        for img_path in image_paths:
            self.model.predict(
                str(img_path),
                conf=self.config['evaluation']['conf_threshold'],
                verbose=False
            )
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        avg_time = total_time / len(image_paths)
        fps = len(image_paths) / total_time
        
        speed_metrics = {
            'total_time': total_time,
            'avg_time_per_image': avg_time,
            'fps': fps,
            'num_images': len(image_paths)
        }
        
        # Log results
        self.logger.info(f"\nSpeed Metrics:")
        self.logger.info(f"  Total Time: {format_time(total_time)}")
        self.logger.info(f"  Average Time per Image: {avg_time*1000:.2f} ms")
        self.logger.info(f"  Throughput: {fps:.2f} FPS")
        
        return speed_metrics
    
    def generate_report(self) -> None:
        """
        Generate comprehensive evaluation report.
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("GENERATING COMPREHENSIVE EVALUATION REPORT")
        self.logger.info("="*80)
        
        # 1. Overall validation metrics
        self.logger.info("\n1. Computing overall validation metrics...")
        val_metrics = self.validate()
        
        # 2. Per-class evaluation
        self.logger.info("\n2. Computing per-class metrics...")
        per_class_df = self.evaluate_per_class()
        
        # 3. Speed benchmark
        self.logger.info("\n3. Benchmarking inference speed...")
        speed_metrics = self.benchmark_speed(num_images=100)
        
        # 4. Generate summary report
        report_path = self.predictions_dir / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TRAFFIC SIGN DETECTION - EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("1. OVERALL METRICS\n")
            f.write("-"*80 + "\n")
            f.write(f"mAP@0.5: {val_metrics['mAP50']:.4f}\n")
            f.write(f"mAP@0.5:0.95: {val_metrics['mAP50-95']:.4f}\n")
            f.write(f"Precision: {val_metrics['precision']:.4f}\n")
            f.write(f"Recall: {val_metrics['recall']:.4f}\n\n")
            
            if len(per_class_df) > 0:
                f.write("2. PER-CLASS PERFORMANCE\n")
                f.write("-"*80 + "\n")
                f.write(f"Best Class: {per_class_df.iloc[0]['Class Name']} ")
                f.write(f"(AP@0.5 = {per_class_df.iloc[0]['AP@0.5']:.4f})\n")
                f.write(f"Worst Class: {per_class_df.iloc[-1]['Class Name']} ")
                f.write(f"(AP@0.5 = {per_class_df.iloc[-1]['AP@0.5']:.4f})\n")
                f.write(f"Average AP@0.5: {per_class_df['AP@0.5'].mean():.4f}\n\n")
            
            if speed_metrics:
                f.write("3. INFERENCE SPEED\n")
                f.write("-"*80 + "\n")
                f.write(f"Average Time: {speed_metrics['avg_time_per_image']*1000:.2f} ms/image\n")
                f.write(f"Throughput: {speed_metrics['fps']:.2f} FPS\n\n")
            
            f.write("="*80 + "\n")
        
        self.logger.info(f"\n✓ Evaluation report saved to: {report_path}")
        
        print("\n" + "="*80)
        print("✓ EVALUATION COMPLETED!")
        print("="*80)
        print(f"Report saved to: {report_path}")
        print(f"Per-class metrics: {self.predictions_dir / 'per_class_metrics.csv'}")
        print("="*80)


def main():
    """Main function to run evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Traffic Sign Detection Model')
    parser.add_argument(
        '--weights',
        type=str,
        default='weights/best.pt',
        help='Path to model weights'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick evaluation (validation only)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = TrafficSignEvaluator(weights_path=args.weights)
        
        if args.quick:
            # Quick evaluation - validation only
            evaluator.validate()
        else:
            # Comprehensive evaluation
            evaluator.generate_report()
        
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
