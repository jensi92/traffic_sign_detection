"""
GTSRB Traffic Sign Detection - Streamlit Web Application

This module provides an interactive web interface for:
- Uploading traffic sign images
- Real-time detection visualization
- Model information and statistics
- Batch processing capabilities
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import io
import tempfile
from ultralytics import YOLO

from utils import load_config, load_class_names, draw_boxes


# Page configuration
st.set_page_config(
    page_title="Traffic Sign Detection",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF4B4B;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .detection-box {
        border: 2px solid #FF4B4B;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f0f2f6;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(weights_path: str):
    """Load YOLO model (cached)."""
    try:
        model = YOLO(weights_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_app_config():
    """Load configuration (cached)."""
    return load_config()


@st.cache_data
def load_classes():
    """Load class names (cached)."""
    return load_class_names()


def predict_image(model, image, conf_threshold, iou_threshold):
    """
    Run prediction on an image.
    
    Args:
        model: YOLO model
        image: Input image (numpy array)
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
        
    Returns:
        Tuple of (result_image, predictions_dict)
    """
    # Run inference
    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )[0]
    
    # Extract predictions
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    labels = results.boxes.cls.cpu().numpy().astype(int)
    
    predictions = {
        'boxes': boxes.tolist(),
        'scores': scores.tolist(),
        'labels': labels.tolist(),
        'num_detections': len(boxes)
    }
    
    # Draw boxes on image
    class_names = load_classes()
    result_image = draw_boxes(
        image,
        boxes.tolist(),
        labels.tolist(),
        scores.tolist(),
        class_names,
        conf_threshold
    )
    
    return result_image, predictions


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🚦 Traffic Sign Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a traffic sign image and let AI detect and classify it!</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        weights_path = st.text_input(
            "Model Weights Path",
            value="weights/best.pt",
            help="Path to trained YOLOv8 model weights"
        )
        
        # Load model
        if Path(weights_path).exists():
            model = load_model(weights_path)
            if model:
                st.success("✓ Model loaded successfully!")
            else:
                st.error("✗ Failed to load model")
                return
        else:
            st.error(f"✗ Model weights not found at: {weights_path}")
            st.info("Please train the model first by running train_model.py")
            return
        
        st.divider()
        
        # Detection parameters
        st.subheader("Detection Parameters")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum confidence score for detections"
        )
        
        iou_threshold = st.slider(
            "IoU Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="IoU threshold for Non-Maximum Suppression"
        )
        
        st.divider()
        
        # Model info
        st.subheader("📊 Model Info")
        config = load_app_config()
        class_names = load_classes()
        
        st.info(f"""
        **Architecture:** YOLOv8  
        **Classes:** {len(class_names)}  
        **Input Size:** {config['model']['img_size']}px
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a traffic sign image...",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload an image containing traffic signs"
        )
        
        # Sample images option
        use_sample = st.checkbox("Or use a sample image from test set")
        
        if use_sample and not uploaded_file:
            # Get a random test image
            test_dir = Path(config['data']['root_dir']) / 'Test'
            if test_dir.exists():
                test_images = list(test_dir.glob('*.png'))[:10]
                if test_images:
                    selected_sample = st.selectbox(
                        "Select a sample image:",
                        test_images,
                        format_func=lambda x: x.name
                    )
                    # Read selected sample
                    image = cv2.imread(str(selected_sample))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    st.image(image, caption="Sample Image", use_container_width=True)
                    uploaded_file = "sample"  # Flag to process
                else:
                    st.warning("No sample images found in Test directory")
            else:
                st.warning("Test directory not found")
        
        elif uploaded_file:
            # Display uploaded image
            image = Image.open(uploaded_file)
            image = np.array(image)
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        else:
            st.info("👆 Please upload an image or select a sample image")
            image = None
    
    with col2:
        st.header("🎯 Detection Results")
        
        if (uploaded_file and image is not None) or (use_sample and 'selected_sample' in locals()):
            # Get image to process
            if use_sample and 'selected_sample' in locals():
                process_image = cv2.imread(str(selected_sample))
                process_image = cv2.cvtColor(process_image, cv2.COLOR_BGR2RGB)
            else:
                process_image = image
            
            # Run detection button
            if st.button("🔍 Detect Traffic Signs", type="primary", use_container_width=True):
                with st.spinner("Detecting traffic signs..."):
                    # Run prediction
                    result_image, predictions = predict_image(
                        model,
                        process_image,
                        conf_threshold,
                        iou_threshold
                    )
                    
                    # Display result
                    st.image(result_image, caption="Detection Result", use_container_width=True)
                    
                    # Display detection statistics
                    num_detections = predictions['num_detections']
                    
                    if num_detections > 0:
                        st.success(f"✓ Detected {num_detections} traffic sign(s)!")
                        
                        # Show detailed detections
                        st.subheader("Detected Signs:")
                        
                        for i, (box, score, label) in enumerate(
                            zip(
                                predictions['boxes'],
                                predictions['scores'],
                                predictions['labels']
                            ),
                            1
                        ):
                            class_name = class_names[label]
                            
                            with st.expander(f"Detection {i}: {class_name} ({score:.2%})"):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Class", class_name)
                                    st.metric("Confidence", f"{score:.2%}")
                                with col_b:
                                    st.metric("Class ID", label)
                                    st.write(f"**Bounding Box:**")
                                    st.write(f"x1={box[0]:.0f}, y1={box[1]:.0f}")
                                    st.write(f"x2={box[2]:.0f}, y2={box[3]:.0f}")
                        
                        # Download button for result
                        result_pil = Image.fromarray(result_image)
                        buf = io.BytesIO()
                        result_pil.save(buf, format='PNG')
                        
                        st.download_button(
                            label="📥 Download Result",
                            data=buf.getvalue(),
                            file_name="detection_result.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    else:
                        st.warning("No traffic signs detected in the image.")
                        st.info("Try adjusting the confidence threshold in the sidebar.")
        else:
            st.info("Upload an image to see detection results here")
    
    # Information section
    st.divider()
    
    st.header("ℹ️ About This System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 What it does
        This system uses YOLOv8 (You Only Look Once) deep learning model 
        to detect and classify traffic signs in images. It can identify 
        43 different types of traffic signs from the German Traffic Sign 
        Recognition Benchmark (GTSRB) dataset.
        """)
    
    with col2:
        st.markdown("""
        ### 🧠 How it works
        1. **Image Upload**: Upload a traffic sign image
        2. **Preprocessing**: Image is resized and normalized
        3. **Detection**: YOLOv8 detects sign locations
        4. **Classification**: Each detection is classified
        5. **Visualization**: Results shown with bounding boxes
        """)
    
    with col3:
        st.markdown("""
        ### 🏷️ Detectable Signs
        The model can detect 43 traffic sign classes including:
        - Speed limits (20-120 km/h)
        - Warning signs (curves, pedestrians, etc.)
        - Regulatory signs (stop, yield, no entry, etc.)
        - Informational signs (roundabout, keep right, etc.)
        """)
    
    # Class reference
    with st.expander("📋 View All Traffic Sign Classes"):
        st.subheader("Complete List of 43 Traffic Sign Classes")
        
        # Display in 3 columns
        classes_per_col = len(class_names) // 3 + 1
        col1, col2, col3 = st.columns(3)
        
        for idx, class_name in enumerate(class_names):
            if idx < classes_per_col:
                with col1:
                    st.write(f"**{idx}.** {class_name}")
            elif idx < classes_per_col * 2:
                with col2:
                    st.write(f"**{idx}.** {class_name}")
            else:
                with col3:
                    st.write(f"**{idx}.** {class_name}")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🚦 Traffic Sign Detection System | Powered by YOLOv8 & Streamlit</p>
        <p>Built for the German Traffic Sign Recognition Benchmark (GTSRB)</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
