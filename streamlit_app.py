import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config and styling
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .success-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #D1E7DD;
        color: #0F5132;
        border: 1px solid #BADBCC;
    }
    .error-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #F8D7DA;
        color: #842029;
        border: 1px solid #F5C2C7;
    }
    .info-box {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #DEE2E6;
        margin: 1rem 0;
    }
    .metrics-container {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #DEE2E6;
        flex: 1;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model with caching
@st.cache_resource(show_spinner="Loading the AI model...")
def load_model():
    try:
        model_path = 'brain_tumor_cnn_model.h5'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully!")
            return model
        else:
            logger.error(f"Model file not found: {model_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

# Image preprocessing with caching
@st.cache_data
def preprocess_image(img):
    try:
        img = img.resize((150, 150))
        img_array = np.array(img)
        
        # Convert to RGB if grayscale
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        
        # Normalize
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, None
    except Exception as e:
        return None, str(e)

def preprocess_image(img):
    img = img.resize((150, 150))
    img_array = np.array(img)
    
    # Convert to RGB if grayscale
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    # Normalize
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Page header
st.title("🧠 Brain Tumor Detection")
st.markdown("""
    This application uses a Convolutional Neural Network (CNN) to detect brain tumors in MRI scans.
    Upload your MRI scan to get instant results!
""")

# Header section with project information
st.markdown("""
    <div class='info-box'>
        <h1 style='text-align: center;'>🧠 Brain Tumor Detection AI</h1>
        <p style='text-align: center;'>Advanced CNN-based detection of brain tumors in MRI scans</p>
    </div>
""", unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 📤 Upload MRI Scan")
    uploaded_file = st.file_uploader(
        "Choose an MRI scan image...", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Supported formats: JPG, JPEG, PNG, BMP"
    )
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded MRI Scan', use_column_width=True)
            
            # Add analyze button
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner('AI is analyzing the image...'):
                    # Load model
                    model = load_model()
                    
                    if model is None:
                        st.error("⚠️ Model not found. Please contact support.")
                    else:
                        # Preprocess and predict
                        processed_image, error = preprocess_image(image)
                        
                        if error:
                            st.error(f"⚠️ Error processing image: {error}")
                        else:
                            start_time = time.time()
                            prediction = model.predict(processed_image, verbose=0)
                            end_time = time.time()
                            
                            probability = float(prediction[0][0])
                            processing_time = end_time - start_time
                            
                            # Display results
                            st.markdown("### 📊 Analysis Results")
                            
                            # Result cards using columns
                            metrics_cols = st.columns(2)
                            
                            # Main result
                            result = "Tumor Detected" if probability > 0.5 else "No Tumor Detected"
                            confidence = probability if probability > 0.5 else 1 - probability
                            
                            with metrics_cols[0]:
                                if result == "Tumor Detected":
                                    st.markdown(
                                        f"""<div class='error-message'>
                                            <h3 style='margin:0'>🔴 {result}</h3>
                                            <p style='margin:0'>Confidence: {confidence:.2%}</p>
                                        </div>""",
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(
                                        f"""<div class='success-message'>
                                            <h3 style='margin:0'>✅ {result}</h3>
                                            <p style='margin:0'>Confidence: {confidence:.2%}</p>
                                        </div>""",
                                        unsafe_allow_html=True
                                    )
                            
                            # Additional metrics
                            with metrics_cols[1]:
                                st.markdown(
                                    f"""<div class='info-box'>
                                        <p><b>Analysis Time:</b> {processing_time:.2f}s</p>
                                        <p><b>Raw Probability:</b> {probability:.4f}</p>
                                    </div>""",
                                    unsafe_allow_html=True
                                )
                            
                            # Technical details expandable section
                            with st.expander("🔧 Technical Details"):
                                st.markdown(f"""
                                    - **Model Architecture:** CNN
                                    - **Input Shape:** {processed_image.shape}
                                    - **Preprocessing Time:** {processing_time:.3f}s
                                    - **Confidence Score:** {confidence:.4f}
                                    """)
        
        except Exception as e:
            st.error(f"⚠️ Error processing image: {str(e)}")

# Instructions and Information
with col2:
    if uploaded_file is None:
        st.markdown("### 🔍 How It Works")
        st.markdown("""
        <div class='info-box'>
            <ol>
                <li>📤 Upload your MRI scan image</li>
                <li>🤖 Advanced AI analyzes the scan</li>
                <li>📊 Get detailed analysis results</li>
                <li>📋 Review confidence scores</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Supported Formats")
        st.markdown("""
        <div class='info-box'>
            <ul>
                <li>JPG/JPEG</li>
                <li>PNG</li>
                <li>BMP</li>
            </ul>
            <p><small>Maximum file size: 5MB</small></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ⚕️ Medical Disclaimer")
        st.markdown("""
        <div class='info-box' style='background-color: #FFF3CD; border-color: #FFE69C;'>
            <p><strong>Important Notice:</strong></p>
            <p>This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis.</p>
            <p>Always consult with qualified healthcare professionals for medical advice.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Developed with ❤️ using TensorFlow and Streamlit</p>
</div>
""", unsafe_allow_html=True)
