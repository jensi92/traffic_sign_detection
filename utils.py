"""
GTSRB Traffic Sign Detection - Utilities Module

This module provides utility functions for:
- Logging and error handling
- Visualization (bounding boxes, detection results)
- Performance metrics computation
- File I/O operations
"""

import os
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict, Any
import yaml
from datetime import datetime


class Logger:
    """Custom logger for training and evaluation."""
    
    def __init__(self, log_dir: str, name: str = "traffic_sign_detection"):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save log files
            name: Logger name
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # File handler
        log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise ValueError(f"Error loading config file: {e}")


def load_class_names(class_names_path: str = "class_names.txt") -> List[str]:
    """
    Load class names from text file.
    
    Args:
        class_names_path: Path to class names file
        
    Returns:
        List of class names
    """
    try:
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    except Exception as e:
        raise ValueError(f"Error loading class names: {e}")


def draw_boxes(
    image: np.ndarray,
    boxes: List[List[float]],
    labels: List[int],
    scores: List[float],
    class_names: List[str],
    conf_threshold: float = 0.25
) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        image: Input image (BGR format)
        boxes: List of bounding boxes [x1, y1, x2, y2]
        labels: List of class labels
        scores: List of confidence scores
        class_names: List of class names
        conf_threshold: Confidence threshold for display
        
    Returns:
        Image with drawn boxes
    """
    img = image.copy()
    
    # Generate random colors for each class
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)
    
    for box, label, score in zip(boxes, labels, scores):
        if score < conf_threshold:
            continue
        
        # Convert box coordinates to integers
        x1, y1, x2, y2 = map(int, box)
        
        # Get color for this class
        color = tuple(map(int, colors[label]))
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        label_text = f"{class_names[label]}: {score:.2f}"
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            img,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            img,
            label_text,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    return img


def visualize_predictions(
    image_path: str,
    predictions: Dict[str, Any],
    class_names: List[str],
    save_path: str = None,
    conf_threshold: float = 0.25
) -> None:
    """
    Visualize predictions on an image.
    
    Args:
        image_path: Path to input image
        predictions: Dictionary containing boxes, labels, and scores
        class_names: List of class names
        save_path: Path to save visualization (optional)
        conf_threshold: Confidence threshold for display
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Draw boxes
    result_img = draw_boxes(
        image,
        predictions['boxes'],
        predictions['labels'],
        predictions['scores'],
        class_names,
        conf_threshold
    )
    
    # Convert BGR to RGB for matplotlib
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    # Display
    plt.figure(figsize=(12, 8))
    plt.imshow(result_img_rgb)
    plt.axis('off')
    plt.title(f"Traffic Sign Detection Results\n{Path(image_path).name}")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: str = None,
    normalize: bool = True
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        save_path: Path to save plot
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / (
            confusion_matrix.sum(axis=1, keepdims=True) + 1e-10
        )
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(
        confusion_matrix,
        annot=False,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Value' if normalize else 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.xticks(rotation=90, ha='right', fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: str = None
) -> None:
    """
    Plot training curves (loss, mAP, etc.).
    
    Args:
        metrics: Dictionary containing training metrics
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss
    if 'train_loss' in metrics:
        axes[0, 0].plot(metrics['train_loss'], label='Train Loss')
        if 'val_loss' in metrics:
            axes[0, 0].plot(metrics['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Plot mAP
    if 'mAP50' in metrics:
        axes[0, 1].plot(metrics['mAP50'], label='mAP@0.5')
        if 'mAP50-95' in metrics:
            axes[0, 1].plot(metrics['mAP50-95'], label='mAP@0.5:0.95')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].set_title('Mean Average Precision')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Plot precision and recall
    if 'precision' in metrics and 'recall' in metrics:
        axes[1, 0].plot(metrics['precision'], label='Precision')
        axes[1, 0].plot(metrics['recall'], label='Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Precision and Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Plot learning rate
    if 'learning_rate' in metrics:
        axes[1, 1].plot(metrics['learning_rate'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    plt.show()


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes.
    
    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    # Get coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Compute intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    intersection_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    
    # Compute union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    # Compute IoU
    iou = intersection_area / (union_area + 1e-10)
    
    return iou


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def create_directory_structure(base_dir: str) -> Dict[str, Path]:
    """
    Create directory structure for the project.
    
    Args:
        base_dir: Base directory path
        
    Returns:
        Dictionary of created paths
    """
    base_path = Path(base_dir)
    
    dirs = {
        'base': base_path,
        'runs': base_path / 'runs',
        'weights': base_path / 'weights',
        'predictions': base_path / 'predictions',
        'logs': base_path / 'logs',
        'dataset': base_path / 'dataset_yolo',
        'dataset_images': base_path / 'dataset_yolo' / 'images',
        'dataset_labels': base_path / 'dataset_yolo' / 'labels',
        'train_images': base_path / 'dataset_yolo' / 'images' / 'train',
        'val_images': base_path / 'dataset_yolo' / 'images' / 'val',
        'train_labels': base_path / 'dataset_yolo' / 'labels' / 'train',
        'val_labels': base_path / 'dataset_yolo' / 'labels' / 'val',
    }
    
    # Create all directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Load config
    config = load_config()
    print(f"Config loaded: {len(config)} sections")
    
    # Load class names
    class_names = load_class_names()
    print(f"Loaded {len(class_names)} class names")
    
    # Create directory structure
    dirs = create_directory_structure(".")
    print(f"Created {len(dirs)} directories")
    
    print("Utilities test completed!")
