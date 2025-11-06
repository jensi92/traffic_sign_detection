"""
GTSRB Traffic Sign Detection - Prediction Module

This module provides inference capabilities:
- Single image prediction
- Batch prediction on multiple images
- Video prediction (frame-by-frame)
- Webcam real-time detection
- Result visualization and saving
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple
import time

from utils import (
    Logger, load_config, load_class_names,
    draw_boxes, visualize_predictions, format_time
)


class TrafficSignPredictor:
    """Predictor class for traffic sign detection."""
    
    def __init__(
        self,
        weights_path: str,
        config_path: str = "config.yaml"
    ):
        """
        Initialize the predictor.
        
        Args:
            weights_path: Path to trained model weights
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = Logger(
            self.config['output']['logs_dir'],
            name="prediction"
        )
        self.class_names = load_class_names()
        
        # Setup paths
        self.predictions_dir = Path(self.config['output']['predictions_dir'])
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        
        # Check weights file
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        # Load model
        self.logger.info(f"Loading model from: {weights_path}")
        self.model = YOLO(weights_path)
        self.logger.info("Model loaded successfully")
        
        # Get thresholds from config
        self.conf_threshold = self.config['evaluation']['conf_threshold']
        self.iou_threshold = self.config['evaluation']['iou_threshold']
        self.max_det = self.config['evaluation']['max_det']
    
    def predict_image(
        self,
        image_path: str,
        save_result: bool = True,
        visualize: bool = True
    ) -> Dict[str, Any]:
        """
        Predict traffic signs in a single image.
        
        Args:
            image_path: Path to input image
            save_result: Whether to save the result image
            visualize: Whether to display the result
            
        Returns:
            Dictionary with predictions
        """
        self.logger.info(f"Predicting on image: {image_path}")
        
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            device=self.config['training']['device'],
            verbose=False
        )[0]  # Get first result
        
        # Extract predictions
        boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        scores = results.boxes.conf.cpu().numpy()
        labels = results.boxes.cls.cpu().numpy().astype(int)
        
        predictions = {
            'boxes': boxes.tolist(),
            'scores': scores.tolist(),
            'labels': labels.tolist(),
            'num_detections': len(boxes)
        }
        
        # Log results
        self.logger.info(f"Detected {len(boxes)} traffic signs")
        for box, score, label in zip(boxes, scores, labels):
            self.logger.info(
                f"  {self.class_names[label]}: {score:.3f} "
                f"at [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]"
            )
        
        # Visualize and save if requested
        if visualize or save_result:
            save_path = None
            if save_result:
                save_path = self.predictions_dir / f"pred_{Path(image_path).name}"
            
            visualize_predictions(
                image_path,
                predictions,
                self.class_names,
                save_path,
                self.conf_threshold
            )
        
        return predictions
    
    def predict_batch(
        self,
        image_paths: List[str],
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Predict traffic signs in multiple images.
        
        Args:
            image_paths: List of image paths
            save_results: Whether to save result images
            
        Returns:
            List of prediction dictionaries
        """
        self.logger.info(f"Running batch prediction on {len(image_paths)} images")
        
        all_predictions = []
        start_time = time.time()
        
        for img_path in image_paths:
            try:
                predictions = self.predict_image(
                    img_path,
                    save_result=save_results,
                    visualize=False
                )
                all_predictions.append({
                    'image_path': img_path,
                    'predictions': predictions
                })
            except Exception as e:
                self.logger.error(f"Error predicting on {img_path}: {e}")
                all_predictions.append({
                    'image_path': img_path,
                    'predictions': None,
                    'error': str(e)
                })
        
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / len(image_paths)
        
        self.logger.info(f"\nBatch prediction completed:")
        self.logger.info(f"  Total time: {format_time(elapsed_time)}")
        self.logger.info(f"  Average time: {avg_time*1000:.2f} ms/image")
        self.logger.info(f"  Throughput: {len(image_paths)/elapsed_time:.2f} FPS")
        
        return all_predictions
    
    def predict_folder(
        self,
        folder_path: str,
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Predict traffic signs in all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            save_results: Whether to save result images
            
        Returns:
            List of prediction dictionaries
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(folder.glob(f'*{ext}'))
            image_paths.extend(folder.glob(f'*{ext.upper()}'))
        
        image_paths = [str(p) for p in image_paths]
        
        self.logger.info(f"Found {len(image_paths)} images in {folder_path}")
        
        if len(image_paths) == 0:
            self.logger.warning("No images found in folder")
            return []
        
        return self.predict_batch(image_paths, save_results)
    
    def predict_video(
        self,
        video_path: str,
        output_path: str = None,
        display: bool = True
    ) -> None:
        """
        Predict traffic signs in a video.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            display: Whether to display the video
        """
        self.logger.info(f"Processing video: {video_path}")
        
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Video info: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Setup output video writer if requested
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run inference on frame
                results = self.model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    max_det=self.max_det,
                    device=self.config['training']['device'],
                    verbose=False
                )[0]
                
                # Draw results on frame
                if len(results.boxes) > 0:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    scores = results.boxes.conf.cpu().numpy()
                    labels = results.boxes.cls.cpu().numpy().astype(int)
                    
                    frame = draw_boxes(
                        frame,
                        boxes.tolist(),
                        labels.tolist(),
                        scores.tolist(),
                        self.class_names,
                        self.conf_threshold
                    )
                
                # Write frame to output video
                if out:
                    out.write(frame)
                
                # Display frame
                if display:
                    cv2.imshow('Traffic Sign Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                
                # Log progress every 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed
                    self.logger.info(
                        f"Processed {frame_count}/{total_frames} frames "
                        f"({fps_actual:.2f} FPS)"
                    )
        
        finally:
            cap.release()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()
        
        elapsed_time = time.time() - start_time
        avg_fps = frame_count / elapsed_time
        
        self.logger.info(f"\nVideo processing completed:")
        self.logger.info(f"  Total frames: {frame_count}")
        self.logger.info(f"  Total time: {format_time(elapsed_time)}")
        self.logger.info(f"  Average FPS: {avg_fps:.2f}")
        
        if output_path:
            self.logger.info(f"  Output saved to: {output_path}")
    
    def predict_webcam(self) -> None:
        """
        Run real-time traffic sign detection on webcam feed.
        
        Press 'q' to quit, 's' to save current frame.
        """
        self.logger.info("Starting webcam detection (Press 'q' to quit, 's' to save)")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Could not open webcam")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run inference
                results = self.model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    max_det=self.max_det,
                    device=self.config['training']['device'],
                    verbose=False
                )[0]
                
                # Draw results
                if len(results.boxes) > 0:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    scores = results.boxes.conf.cpu().numpy()
                    labels = results.boxes.cls.cpu().numpy().astype(int)
                    
                    frame = draw_boxes(
                        frame,
                        boxes.tolist(),
                        labels.tolist(),
                        scores.tolist(),
                        self.class_names,
                        self.conf_threshold
                    )
                
                # Calculate and display FPS
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Display frame
                cv2.imshow('Traffic Sign Detection - Webcam', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    save_path = self.predictions_dir / f"webcam_{frame_count}.jpg"
                    cv2.imwrite(str(save_path), frame)
                    self.logger.info(f"Frame saved to: {save_path}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        self.logger.info(f"Webcam detection stopped after {frame_count} frames")


def main():
    """Main function to run predictions."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Traffic Sign Detection Prediction')
    parser.add_argument(
        '--weights',
        type=str,
        default='weights/best.pt',
        help='Path to model weights'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source: image path, folder path, video path, or "webcam"'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save prediction results'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not display results'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = TrafficSignPredictor(weights_path=args.weights)
        
        # Determine source type and run prediction
        source = args.source.lower()
        
        if source == 'webcam':
            # Webcam detection
            predictor.predict_webcam()
        
        elif Path(args.source).is_file():
            # Check if video or image
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            if Path(args.source).suffix.lower() in video_extensions:
                # Video prediction
                output_path = None
                if args.save:
                    output_path = str(
                        predictor.predictions_dir /
                        f"pred_{Path(args.source).name}"
                    )
                predictor.predict_video(
                    args.source,
                    output_path,
                    display=not args.no_display
                )
            else:
                # Image prediction
                predictor.predict_image(
                    args.source,
                    save_result=args.save,
                    visualize=not args.no_display
                )
        
        elif Path(args.source).is_dir():
            # Folder prediction
            predictor.predict_folder(args.source, save_results=args.save)
        
        else:
            print(f"Invalid source: {args.source}")
            print("Source must be: image path, folder path, video path, or 'webcam'")
        
        print("\n✓ Prediction completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during prediction: {e}")
        raise


if __name__ == "__main__":
    main()
