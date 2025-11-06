# 🚀 Quick Start Guide - Traffic Sign Detection

## Step-by-Step Instructions

### Prerequisites Check
- [ ] Python 3.8+ installed
- [ ] GTSRB dataset downloaded and extracted in project folder
- [ ] 10GB+ free disk space
- [ ] GPU with CUDA (recommended, but optional)

### 1️⃣ Setup Environment (5 minutes)

```powershell
# Navigate to project directory
cd e:\project

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2️⃣ Prepare Dataset (10 minutes)

```powershell
# Convert GTSRB to YOLO format
python data_preprocessing.py
```

**What happens:**
- Reads Train.csv annotations
- Splits data 80/20 (train/validation)
- Converts bounding boxes to YOLO format
- Creates dataset_yolo/ directory
- Generates dataset.yaml

**Expected output:**
```
Processing training set: 100%
Processing validation set: 100%
✓ Successfully processed 31,367 training images
✓ Successfully processed 7,842 validation images
✓ Dataset ready for training
```

### 3️⃣ Train Model (2-3 hours on GPU)

```powershell
# Start training
python train_model.py
```

**What happens:**
- Downloads YOLOv8n pretrained weights (if needed)
- Trains for 10 epochs with data augmentation
- Saves checkpoints every 10 epochs
- Validates after each epoch
- Saves best model to weights/best.pt

**Monitor progress:**
- Watch terminal for loss/metrics
- Check logs/training_*.log for details
- Or use TensorBoard: `tensorboard --logdir runs/train`

**Expected metrics after training:**
- mAP@0.5: >0.95
- Training time: 2-3 hours (GPU) / 20+ hours (CPU)

### 4️⃣ Evaluate Model (5 minutes)

```powershell
# Run comprehensive evaluation
python evaluate_model.py --weights weights/best.pt
```

**What happens:**
- Validates on validation set
- Computes mAP, precision, recall
- Analyzes per-class performance
- Benchmarks inference speed
- Generates evaluation report

**Output files:**
- predictions/evaluation_report.txt
- predictions/per_class_metrics.csv

### 5️⃣ Test Predictions (Quick)

#### Test on single image:
```powershell
python predict.py --weights weights/best.pt --source Test/00001.png --save
```

#### Test on multiple images:
```powershell
python predict.py --weights weights/best.pt --source Test/ --save
```

**Output:**
- Annotated images saved to predictions/
- Detection results logged to console

### 6️⃣ Launch Web Interface (Instant)

```powershell
# Start Streamlit app
streamlit run app.py
```

**Access at:** http://localhost:8501

**Features:**
- Upload traffic sign images
- Adjust detection thresholds
- View results in real-time
- Download annotated images
- Try sample images

---

## ⚡ One-Command Setup (Advanced)

If you want to automate the entire process:

```powershell
# Create a setup script: setup.ps1
# Run all steps in sequence
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python data_preprocessing.py
python train_model.py
```

---

## 🎯 Quick Commands Reference

### Training
```powershell
# Start training
python train_model.py

# Resume training from checkpoint
python train_model.py  # Automatically resumes if checkpoint exists
```

### Evaluation
```powershell
# Full evaluation
python evaluate_model.py --weights weights/best.pt

# Quick validation only
python evaluate_model.py --weights weights/best.pt --quick
```

### Prediction
```powershell
# Single image
python predict.py --weights weights/best.pt --source image.png --save

# Folder
python predict.py --weights weights/best.pt --source path/to/folder/ --save

# Video
python predict.py --weights weights/best.pt --source video.mp4 --save

# Webcam
python predict.py --weights weights/best.pt --source webcam
```

### Web Interface
```powershell
# Launch Streamlit app
streamlit run app.py

# Launch on specific port
streamlit run app.py --server.port 8080
```

---

## 🔍 Verification Steps

After each step, verify success:

### After Setup:
```powershell
python -c "import torch; import ultralytics; print('✓ Setup OK')"
```

### After Preprocessing:
```powershell
# Check if dataset directory exists
Test-Path dataset_yolo/dataset.yaml
# Should return: True
```

### After Training:
```powershell
# Check if model weights exist
Test-Path weights/best.pt
# Should return: True
```

### After Evaluation:
```powershell
# Check if evaluation report exists
Test-Path predictions/evaluation_report.txt
# Should return: True
```

---

## 🐛 Common Issues & Quick Fixes

### Issue: "CUDA out of memory"
**Fix:** Reduce batch size in config.yaml
```yaml
training:
  batch_size: 8  # Reduce from 16
```

### Issue: "Dataset YAML not found"
**Fix:** Run preprocessing first
```powershell
python data_preprocessing.py
```

### Issue: "No module named 'ultralytics'"
**Fix:** Install dependencies
```powershell
pip install -r requirements.txt
```

### Issue: Slow training on CPU
**Fix:** Training on CPU is very slow. Options:
1. Use GPU (recommended)
2. Use Google Colab with free GPU
3. Reduce epochs in config.yaml (for testing)
4. Use smaller model: yolov8n.pt

---

## 📊 Expected Timeline

| Step | Time (GPU) | Time (CPU) |
|------|------------|------------|
| Setup | 5 min | 5 min |
| Preprocessing | 10 min | 10 min |
| Training | 2-3 hours | 20+ hours |
| Evaluation | 5 min | 15 min |
| **Total** | **~3 hours** | **~21 hours** |

---

## 🎓 Learning Path

If you're new to object detection, follow this order:

1. **Read README.md** - Understand the project
2. **Run preprocessing** - See data preparation
3. **Explore utils.py** - Understand helper functions
4. **Study train_model.py** - Learn training process
5. **Examine predictions** - See model in action
6. **Modify config.yaml** - Experiment with parameters
7. **Review evaluation** - Understand metrics

---

## 🆘 Getting Help

1. **Check logs:** `logs/` directory contains detailed logs
2. **Review README.md:** Comprehensive documentation
3. **Read code comments:** Each file is well-documented
4. **Check configuration:** Verify config.yaml settings

---

## ✅ Success Checklist

Before considering the project complete:

- [ ] Environment setup successful
- [ ] Dataset preprocessed (39,209 samples split)
- [ ] Model trained (100 epochs completed)
- [ ] mAP@0.5 > 0.90 achieved
- [ ] Evaluation report generated
- [ ] Predictions work on test images
- [ ] Web interface launches successfully
- [ ] Can detect traffic signs in real-time

---

## 🎉 Next Steps

After completing the quick start:

1. **Experiment** with different hyperparameters
2. **Try** different YOLO model sizes (n/s/m/l/x)
3. **Test** on your own traffic sign images
4. **Deploy** the model to a cloud service
5. **Optimize** for edge devices
6. **Extend** to detect other objects

---

**Happy Coding! 🚀**
