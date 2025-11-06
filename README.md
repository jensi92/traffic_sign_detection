# 🚦 Traffic Sign Detection System

A complete traffic sign object detection project using YOLOv8 and the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Web Interface](#web-interface)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project implements a state-of-the-art traffic sign detection system using:
- **YOLOv8**: Latest version of the YOLO object detection architecture
- **GTSRB Dataset**: 43 classes of German traffic signs
- **PyTorch**: Deep learning framework
- **Streamlit**: Interactive web interface

The system can detect and classify 43 different types of traffic signs in real-time, including speed limits, warning signs, regulatory signs, and informational signs.

## ✨ Features

- ✅ **Complete Training Pipeline**: From data preprocessing to model deployment
- ✅ **YOLOv8 Architecture**: State-of-the-art object detection
- ✅ **Comprehensive Evaluation**: mAP, Precision, Recall, Confusion Matrix
- ✅ **Multiple Inference Modes**: Single image, batch, video, webcam
- ✅ **Interactive Web Interface**: Streamlit-based UI for easy use
- ✅ **Well-Documented Code**: Extensive comments explaining ML/DL concepts
- ✅ **Modular Design**: Easy to extend and customize
- ✅ **Production Ready**: Error handling, logging, and checkpointing

## 📊 Dataset

### GTSRB (German Traffic Sign Recognition Benchmark)

- **43 classes** of traffic signs
- **~39,000 training images**
- **~12,000 test images**
- Various sizes, lighting conditions, and viewing angles

**Dataset Structure:**
```
project/
├── Train/          # Training images organized by class
│   ├── 0/          # Speed limit (20km/h)
│   ├── 1/          # Speed limit (30km/h)
│   └── ...
├── Test/           # Test images
├── Train.csv       # Training annotations
├── Test.csv        # Test annotations
└── Meta.csv        # Class metadata
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

### Step 1: Clone or Set Up Project

```powershell
# Navigate to your project directory
cd e:\project
```

### Step 2: Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### Step 4: Verify Installation

```powershell
# Test utilities
python utils.py
```

## 🎬 Quick Start

### 1. Preprocess Dataset

Convert GTSRB dataset to YOLO format:

```powershell
python data_preprocessing.py
```

This will:
- Load training data from CSV
- Split into train/validation sets (80/20)
- Convert annotations to YOLO format
- Create `dataset_yolo/` directory structure

**Output:**
```
dataset_yolo/
├── images/
│   ├── train/      # Training images
│   └── val/        # Validation images
├── labels/
│   ├── train/      # Training labels (YOLO format)
│   └── val/        # Validation labels
└── dataset.yaml    # Dataset configuration
```

### 2. Train Model

Train YOLOv8 model:

```powershell
python train_model.py
```

**Training Configuration** (edit `config.yaml` to customize):
- **Epochs**: 10 
- **Batch Size**: 16
- **Image Size**: 416x416
- **Learning Rate**: 0.01
- **Optimizer**: SGD

**Training Features:**
- Automatic mixed precision (AMP) for faster training
- Cosine learning rate scheduling
- Early stopping with patience
- Model checkpointing
- Real-time metrics logging
- TensorBoard visualization

**Expected Training Time:**
- GPU (RTX 3060): ~2-3 hours
- GPU (RTX 3090): ~1-1.5 hours
- CPU: Not recommended (very slow)

### 3. Evaluate Model

Evaluate trained model:

```powershell
# Comprehensive evaluation
python evaluate_model.py --weights weights/best.pt

# Quick validation only
python evaluate_model.py --weights weights/best.pt --quick
```

**Evaluation Metrics:**
- **mAP@0.5**: Mean Average Precision at IoU 0.5
- **mAP@0.5:0.95**: mAP averaged over IoU thresholds 0.5 to 0.95
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **Per-class Performance**: AP for each of 43 classes
- **Inference Speed**: FPS and latency

### 4. Make Predictions

#### Single Image

```powershell
python predict.py --weights weights/best.pt --source path/to/image.png --save
```

#### Folder of Images

```powershell
python predict.py --weights weights/best.pt --source path/to/folder --save
```

#### Video

```powershell
python predict.py --weights weights/best.pt --source path/to/video.mp4 --save
```

#### Webcam (Real-time)

```powershell
python predict.py --weights weights/best.pt --source webcam
```

### 5. Launch Web Interface

Start Streamlit web app:

```powershell
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Web Interface Features:**
- Upload traffic sign images
- Real-time detection visualization
- Adjustable confidence threshold
- Download results
- Sample images from test set
- Complete class reference

## 📁 Project Structure

```
project/
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── class_names.txt            # 43 traffic sign class names
│
├── utils.py                   # Utility functions
├── data_preprocessing.py      # Dataset preprocessing
├── train_model.py            # Training pipeline
├── evaluate_model.py         # Model evaluation
├── predict.py                # Inference script
├── app.py                    # Streamlit web app
│
├── dataset_yolo/             # Preprocessed dataset (created)
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── dataset.yaml
│
├── runs/                     # Training runs (created)
│   └── train/
│       ├── weights/
│       ├── results.png
│       └── ...
│
├── weights/                  # Saved model weights (created)
│   └── best.pt
│
├── predictions/              # Prediction results (created)
├── logs/                     # Log files (created)
│
├── Train/                    # Original GTSRB dataset
├── Test/
├── Train.csv
└── Test.csv
```

## 📚 Usage

### Configuration

Edit `config.yaml` to customize:

```yaml
# Model selection
model:
  architecture: "yolov8n.pt"  # yolov8n/s/m/l/x

# Training parameters
training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.01
  device: "0"  # GPU device or "cpu"

# Data augmentation
augmentation:
  degrees: 10       # Rotation
  translate: 0.1    # Translation
  scale: 0.5        # Scaling
  fliplr: 0.5       # Horizontal flip
```

### Advanced Training

#### Resume Training

```powershell
python train_model.py  # Will resume from last checkpoint if exists
```

#### Custom Configuration

```python
from train_model import TrafficSignTrainer

trainer = TrafficSignTrainer(config_path="custom_config.yaml")
results = trainer.train()
```

### Custom Prediction

```python
from predict import TrafficSignPredictor

predictor = TrafficSignPredictor(weights_path="weights/best.pt")
predictions = predictor.predict_image("image.png")

for i, (box, score, label) in enumerate(zip(
    predictions['boxes'],
    predictions['scores'],
    predictions['labels']
)):
    print(f"Detection {i+1}: Class {label}, Score {score:.3f}")
```

## 🧠 Model Architecture

### YOLOv8 Overview

YOLOv8 (You Only Look Once version 8) is a state-of-the-art object detection model with:

**Backbone**: CSPDarknet
- Feature extraction using Cross Stage Partial networks
- Residual connections for better gradient flow
- Multiple scales of feature maps

**Neck**: PANet (Path Aggregation Network)
- Bottom-up and top-down feature fusion
- Multi-scale feature aggregation
- Enhanced localization capability

**Head**: Decoupled Detection Head
- Separate heads for classification and localization
- Anchor-free detection (no predefined anchor boxes)
- Distribution Focal Loss for better box regression

### Key Improvements over Previous YOLO Versions

1. **Anchor-Free**: No need to define anchor boxes
2. **Task-Aligned Assigner**: Better positive/negative sample assignment
3. **Mosaic Augmentation**: Improved generalization
4. **Auto-Augmentation**: Automatically selects best augmentation policy
5. **Mixed Precision Training**: Faster training with lower memory

### Loss Function

YOLOv8 uses a combination of three losses:

1. **CIoU Loss** (Complete IoU): For bounding box regression
   - Considers overlap, distance, and aspect ratio
   
2. **BCE Loss** (Binary Cross-Entropy): For classification
   - Multi-label classification support
   
3. **DFL Loss** (Distribution Focal Loss): For box distribution
   - Better box boundary localization

### Training Process

1. **Data Loading**: Images loaded with on-the-fly augmentation
2. **Forward Pass**: 
   - Image → Backbone → Neck → Head
   - Predictions: [x, y, w, h, confidence, class_probabilities]
3. **Loss Calculation**: CIoU + BCE + DFL losses
4. **Backward Pass**: Gradients computed via backpropagation
5. **Optimization**: Weights updated using SGD/Adam
6. **Validation**: Periodic evaluation on validation set

## 🎓 Training

### Training Process Explained

#### 1. Data Augmentation

Applied during training to improve generalization:

```python
# Geometric augmentations
- Random rotation (±10°)
- Random translation (±10%)
- Random scaling (0.5-1.5x)
- Horizontal flip (50% chance)

# Color augmentations
- HSV color space jittering
- Random brightness/contrast

# Advanced augmentations
- Mosaic (combining 4 images)
- MixUp (blending images)
```

#### 2. Convolutional Neural Networks (CNNs)

CNNs are the foundation of YOLO:

**Convolutional Layers**:
- Apply learnable filters to extract features
- Early layers detect edges and textures
- Deeper layers detect complex patterns (sign shapes, numbers)

**Pooling Layers**:
- Reduce spatial dimensions
- Provide translation invariance
- Max pooling keeps strongest activations

**Batch Normalization**:
- Normalizes layer inputs
- Stabilizes training
- Allows higher learning rates

**Activation Functions** (SiLU/Swish):
- Introduces non-linearity
- Allows network to learn complex patterns

#### 3. Training Loop

```python
for epoch in range(num_epochs):
    for batch in training_data:
        # Forward pass
        predictions = model(batch_images)
        
        # Compute loss
        loss = compute_loss(predictions, targets)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
    
    # Validate
    if epoch % val_frequency == 0:
        validate_model()
```

#### 4. Learning Rate Scheduling

Uses cosine annealing:
- Starts at initial LR (0.01)
- Gradually decreases following cosine curve
- Allows fine-tuning in later epochs
- Improves final performance

#### 5. Early Stopping

Prevents overfitting:
- Monitors validation mAP
- Stops if no improvement for N epochs (patience=50)
- Saves best weights automatically

### Monitoring Training

#### TensorBoard

```powershell
tensorboard --logdir runs/train
```

View in browser: `http://localhost:6006`

**Metrics to Watch**:
- Training/Validation Loss (should decrease)
- mAP@0.5 (should increase, target >0.9)
- Precision and Recall (should be balanced)
- Learning Rate (follows schedule)

#### Log Files

Check `logs/training_*.log` for detailed information:
- Epoch progress
- Loss values
- Validation metrics
- Training time

## 📈 Evaluation

### Evaluation Metrics Explained

#### 1. Intersection over Union (IoU)

Measures box overlap:

```
IoU = Area of Overlap / Area of Union
```

- IoU > 0.5: Generally considered a good detection
- IoU > 0.75: High-quality detection

#### 2. Precision

```
Precision = True Positives / (True Positives + False Positives)
```

- High precision: Few false alarms
- Important when false positives are costly

#### 3. Recall

```
Recall = True Positives / (True Positives + False Negatives)
```

- High recall: Detects most objects
- Important when missing objects is costly

#### 4. Average Precision (AP)

- Area under Precision-Recall curve
- Computed per class
- Ranges from 0 to 1 (higher is better)

#### 5. Mean Average Precision (mAP)

```
mAP = Average of AP across all classes
```

- **mAP@0.5**: AP at IoU threshold 0.5
- **mAP@0.5:0.95**: Average AP from IoU 0.5 to 0.95 (step 0.05)

### Expected Performance

**Target Metrics** (after 100 epochs):
- mAP@0.5: **>0.95**
- mAP@0.5:0.95: **>0.85**
- Precision: **>0.92**
- Recall: **>0.90**
- Inference Speed: **>50 FPS** (on GPU)

## 🌐 Web Interface

### Starting the App

```powershell
streamlit run app.py
```

### Features

1. **Image Upload**: Drag and drop or browse
2. **Sample Images**: Test with dataset samples
3. **Adjustable Thresholds**: Fine-tune detection sensitivity
4. **Real-time Results**: Instant visualization
5. **Downloadable Results**: Save annotated images
6. **Model Information**: View architecture and stats
7. **Class Reference**: Complete list of 43 classes

### Usage Tips

- Lower confidence threshold to detect more signs (may increase false positives)
- Higher confidence threshold for more certain detections
- Adjust IoU threshold to control overlapping detections
- Use sample images to test before uploading your own

## 🎯 Results

### Sample Detections

The trained model can accurately detect and classify:

**Speed Limit Signs**:
- 20, 30, 50, 60, 70, 80, 100, 120 km/h
- End of speed limit zones

**Warning Signs**:
- Dangerous curves
- Bumpy road
- Slippery road
- Road work
- Pedestrians
- Children crossing
- Wild animals

**Regulatory Signs**:
- Stop
- Yield
- No entry
- No passing
- Priority road

**Informational Signs**:
- Keep right/left
- Roundabout mandatory
- Turn ahead
- Go straight

### Performance Characteristics

**Strengths**:
- ✅ High accuracy on clear, well-lit images
- ✅ Robust to small rotations and scale changes
- ✅ Fast inference (real-time capable)
- ✅ Works with multiple signs in single image

**Challenges**:
- ⚠️ May struggle with heavily occluded signs
- ⚠️ Performance drops in extreme lighting conditions
- ⚠️ Small signs at distance may be missed

## 🔧 Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM) Error

**Solution**: Reduce batch size in `config.yaml`

```yaml
training:
  batch_size: 8  # Reduce from 16
```

#### 2. CUDA Not Available

**Check**:
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

**Solution**: Set device to CPU in config.yaml
```yaml
training:
  device: "cpu"
```

#### 3. Dataset Not Found

**Error**: `Dataset YAML not found`

**Solution**: Run preprocessing first
```powershell
python data_preprocessing.py
```

#### 4. Model Weights Not Found

**Error**: `Weights file not found`

**Solution**: Train model first or check path
```powershell
python train_model.py
```

#### 5. Slow Training

**Solutions**:
- Enable AMP (automatic mixed precision) - already enabled
- Reduce image size in config
- Use smaller model (yolov8n instead of yolov8x)
- Ensure GPU is being used

#### 6. Poor Performance

**Solutions**:
- Train for more epochs
- Increase data augmentation
- Check for class imbalance
- Try different learning rate
- Use larger model (yolov8m or yolov8l)

### Getting Help

- Check log files in `logs/` directory
- Review training curves in TensorBoard
- Verify dataset preprocessing completed successfully
- Ensure all dependencies are installed correctly

## 📝 File Descriptions

### Core Modules

**`utils.py`**: Utility functions
- Logger class for consistent logging
- Configuration loading
- Visualization functions (bounding boxes, confusion matrix)
- Metric computation (IoU, etc.)
- Directory structure management

**`data_preprocessing.py`**: Dataset preparation
- Load GTSRB CSV annotations
- Convert to YOLO format (normalized coordinates)
- Train/validation split with stratification
- Generate dataset.yaml for YOLO
- Dataset statistics and validation

**`train_model.py`**: Training pipeline
- YOLOv8 model initialization
- Training loop with callbacks
- Learning rate scheduling
- Model checkpointing
- Metric logging and visualization

**`evaluate_model.py`**: Model evaluation
- Validation metrics computation
- Per-class performance analysis
- Confusion matrix generation
- Speed benchmarking
- Comprehensive report generation

**`predict.py`**: Inference
- Single image prediction
- Batch processing
- Video processing
- Webcam real-time detection
- Result visualization and saving

**`app.py`**: Web interface
- Streamlit-based UI
- File upload handling
- Real-time detection
- Interactive parameter adjustment
- Result visualization and download

### Configuration Files

**`config.yaml`**: Main configuration
- Dataset paths
- Model architecture selection
- Training hyperparameters
- Augmentation parameters
- Evaluation thresholds

**`class_names.txt`**: Traffic sign classes
- 43 class names in order
- Used for label mapping

**`requirements.txt`**: Python dependencies
- Deep learning frameworks
- Computer vision libraries
- Web interface tools
- Utilities

## 🚀 Future Improvements

Potential enhancements:

1. **Model Improvements**:
   - Ensemble multiple models
   - Test YOLOv9 or YOLOv10
   - Implement model quantization for mobile deployment

2. **Dataset Enhancements**:
   - Add more augmentation techniques
   - Incorporate additional traffic sign datasets
   - Synthetic data generation

3. **Features**:
   - Multi-camera support
   - GPS integration for location-aware detection
   - Sign tracking across video frames
   - Warning system for drivers

4. **Deployment**:
   - Docker containerization
   - REST API for integration
   - Mobile app (iOS/Android)
   - Edge device deployment (Raspberry Pi, Jetson)

5. **Performance**:
   - TensorRT optimization
   - ONNX export for cross-platform
   - Model pruning for faster inference

## 📄 License

This project is created for educational purposes as part of a machine learning demonstration.

## 🙏 Acknowledgments

- **GTSRB Dataset**: J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark
- **YOLOv8**: Ultralytics - https://github.com/ultralytics/ultralytics
- **PyTorch**: Facebook AI Research
- **Streamlit**: Streamlit Inc.

## 📧 Contact

For questions or issues, please check the troubleshooting section above or review the documentation in each Python file.

---

**Built with ❤️ using YOLOv8, PyTorch, and Streamlit**
