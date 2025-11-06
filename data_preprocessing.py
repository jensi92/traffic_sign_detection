"""
GTSRB Traffic Sign Detection - Data Preprocessing Module

This module handles:
- Loading GTSRB dataset from CSV files
- Converting image annotations to YOLO format
- Creating train/validation split
- Data augmentation preprocessing
- Dataset statistics and visualization

The GTSRB (German Traffic Sign Recognition Benchmark) dataset contains
43 classes of traffic signs with varying sizes and lighting conditions.
"""

import os
import pandas as pd
import numpy as np
import cv2
import shutil
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml

from utils import Logger, load_config, load_class_names, create_directory_structure


class GTSRBPreprocessor:
    """Preprocessor for GTSRB dataset to YOLO format."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the preprocessor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = Logger(
            self.config['output']['logs_dir'],
            name="data_preprocessing"
        )
        self.class_names = load_class_names()
        
        # Create directory structure
        self.dirs = create_directory_structure(self.config['data']['root_dir'])
        self.logger.info(f"Initialized preprocessor with {len(self.class_names)} classes")
    
    def load_training_data(self) -> pd.DataFrame:
        """
        Load training data from CSV file.
        
        The CSV contains: Width, Height, Roi.X1, Roi.Y1, Roi.X2, Roi.Y2, ClassId, Path
        
        Returns:
            DataFrame with training annotations
        """
        csv_path = os.path.join(
            self.config['data']['root_dir'],
            self.config['data']['train_csv']
        )
        
        try:
            df = pd.read_csv(csv_path)
            self.logger.info(f"Loaded {len(df)} training samples from {csv_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading training data: {e}")
            raise
    
    def convert_to_yolo_format(
        self,
        img_width: int,
        img_height: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int
    ) -> Tuple[float, float, float, float]:
        """
        Convert bounding box to YOLO format.
        
        YOLO format: [x_center, y_center, width, height] (normalized 0-1)
        Input format: [x1, y1, x2, y2] (pixel coordinates)
        
        Args:
            img_width: Image width
            img_height: Image height
            x1, y1, x2, y2: Bounding box coordinates
            
        Returns:
            Tuple of (x_center, y_center, width, height) normalized
        """
        # Calculate center coordinates and dimensions
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        # Normalize by image dimensions
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        # Clip values to [0, 1] range
        x_center = np.clip(x_center, 0, 1)
        y_center = np.clip(y_center, 0, 1)
        width = np.clip(width, 0, 1)
        height = np.clip(height, 0, 1)
        
        return x_center, y_center, width, height
    
    def process_single_image(
        self,
        row: pd.Series,
        output_img_dir: Path,
        output_label_dir: Path
    ) -> bool:
        """
        Process a single image and create YOLO annotation.
        
        Args:
            row: DataFrame row with image annotation
            output_img_dir: Output directory for images
            output_label_dir: Output directory for labels
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get image path
            img_path = os.path.join(self.config['data']['root_dir'], row['Path'])
            
            # Read image to get actual dimensions
            image = cv2.imread(img_path)
            if image is None:
                self.logger.warning(f"Could not read image: {img_path}")
                return False
            
            actual_height, actual_width = image.shape[:2]
            
            # Convert bounding box to YOLO format
            x_center, y_center, width, height = self.convert_to_yolo_format(
                actual_width,
                actual_height,
                row['Roi.X1'],
                row['Roi.Y1'],
                row['Roi.X2'],
                row['Roi.Y2']
            )
            
            # Create label line: class_id x_center y_center width height
            label_line = f"{row['ClassId']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            
            # Create output paths
            img_name = Path(row['Path']).name
            label_name = Path(row['Path']).stem + '.txt'
            
            output_img_path = output_img_dir / img_name
            output_label_path = output_label_dir / label_name
            
            # Copy image
            shutil.copy2(img_path, output_img_path)
            
            # Write label file
            with open(output_label_path, 'w') as f:
                f.write(label_line)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing image {row['Path']}: {e}")
            return False
    
    def split_dataset(
        self,
        df: pd.DataFrame,
        val_split: float = 0.2,
        random_seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into training and validation sets.
        
        Uses stratified split to maintain class distribution.
        
        Args:
            df: DataFrame with all samples
            val_split: Validation split ratio
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df)
        """
        self.logger.info(f"Splitting dataset with {val_split:.0%} validation split")
        
        # Stratified split by class to maintain class distribution
        train_df, val_df = train_test_split(
            df,
            test_size=val_split,
            random_state=random_seed,
            stratify=df['ClassId']
        )
        
        self.logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
        
        # Log class distribution
        train_dist = train_df['ClassId'].value_counts().sort_index()
        val_dist = val_df['ClassId'].value_counts().sort_index()
        
        self.logger.info(f"Train class distribution:\n{train_dist}")
        self.logger.info(f"Val class distribution:\n{val_dist}")
        
        return train_df, val_df
    
    def process_dataset(self) -> None:
        """
        Process the entire GTSRB dataset.
        
        Steps:
        1. Load training data
        2. Split into train/val
        3. Convert images and annotations to YOLO format
        4. Save dataset configuration
        """
        self.logger.info("Starting dataset preprocessing...")
        
        # Load data
        df = self.load_training_data()
        
        # Split dataset
        val_split = self.config['validation']['split_ratio']
        random_seed = self.config['validation']['random_seed']
        train_df, val_df = self.split_dataset(df, val_split, random_seed)
        
        # Process training set
        self.logger.info("Processing training set...")
        train_success = 0
        for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train"):
            if self.process_single_image(
                row,
                self.dirs['train_images'],
                self.dirs['train_labels']
            ):
                train_success += 1
        
        self.logger.info(f"Successfully processed {train_success}/{len(train_df)} training images")
        
        # Process validation set
        self.logger.info("Processing validation set...")
        val_success = 0
        for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Val"):
            if self.process_single_image(
                row,
                self.dirs['val_images'],
                self.dirs['val_labels']
            ):
                val_success += 1
        
        self.logger.info(f"Successfully processed {val_success}/{len(val_df)} validation images")
        
        # Create dataset YAML for YOLO
        self.create_dataset_yaml()
        
        # Print statistics
        self.print_dataset_statistics(train_df, val_df)
        
        self.logger.info("Dataset preprocessing completed!")
    
    def create_dataset_yaml(self) -> None:
        """
        Create dataset configuration YAML file for YOLO training.
        
        This file defines paths and class names for YOLO model.
        """
        dataset_yaml = {
            'path': str(self.dirs['dataset'].absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        yaml_path = self.dirs['dataset'] / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f, sort_keys=False)
        
        self.logger.info(f"Created dataset YAML at {yaml_path}")
    
    def print_dataset_statistics(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
    ) -> None:
        """
        Print dataset statistics.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
        """
        print("\n" + "="*80)
        print("DATASET STATISTICS")
        print("="*80)
        
        print(f"\nTotal Classes: {len(self.class_names)}")
        print(f"Training Samples: {len(train_df)}")
        print(f"Validation Samples: {len(val_df)}")
        print(f"Total Samples: {len(train_df) + len(val_df)}")
        
        # Class distribution
        print(f"\nClass Distribution (Top 10 most common):")
        class_dist = train_df['ClassId'].value_counts().head(10)
        for class_id, count in class_dist.items():
            print(f"  Class {class_id} ({self.class_names[class_id]}): {count} samples")
        
        # Bounding box statistics
        print(f"\nBounding Box Statistics:")
        widths = train_df['Roi.X2'] - train_df['Roi.X1']
        heights = train_df['Roi.Y2'] - train_df['Roi.Y1']
        
        print(f"  Average Width: {widths.mean():.1f} px")
        print(f"  Average Height: {heights.mean():.1f} px")
        print(f"  Min Width: {widths.min()} px")
        print(f"  Max Width: {widths.max()} px")
        print(f"  Min Height: {heights.min()} px")
        print(f"  Max Height: {heights.max()} px")
        
        print("\n" + "="*80)


def main():
    """Main function to run data preprocessing."""
    try:
        # Initialize preprocessor
        preprocessor = GTSRBPreprocessor()
        
        # Process dataset
        preprocessor.process_dataset()
        
        print("\n✓ Data preprocessing completed successfully!")
        print(f"✓ Dataset ready for training in: {preprocessor.dirs['dataset']}")
        print(f"✓ Dataset YAML: {preprocessor.dirs['dataset'] / 'dataset.yaml'}")
        
    except Exception as e:
        print(f"\n✗ Error during preprocessing: {e}")
        raise


if __name__ == "__main__":
    main()
