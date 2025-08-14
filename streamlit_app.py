import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import time
import logging

# Configure logging
logging.basicCon# Header section with project information
st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h1 style='font-size: 3rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #D76D77, #5A1C71); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            🧠 Brain Tumor Detection
        </h1>
        <p style='font-size: 1.2rem; color: rgba(255, 255, 255, 0.8); margin-top: 1rem;'>
            Advanced AI-Powered Detection System for Brain Tumor Analysis
        </p>
    </div>
    <div style='height: 2px; background: linear-gradient(90deg, transparent, #D76D77, transparent); margin: 1rem 0;'></div>
""", unsafe_allow_html=True)el=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config and styling
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS with new theme
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        background: linear-gradient(135deg, #5A1C71, #3A1C71);
    }
    .success-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(209, 231, 221, 0.1);
        color: #D76D77;
        border: 1px solid rgba(215, 109, 119, 0.2);
        backdrop-filter: blur(10px);
    }
    .error-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(215, 109, 119, 0.1);
        color: #D76D77;
        border: 1px solid rgba(215, 109, 119, 0.2);
        backdrop-filter: blur(10px);
    }
    .info-box {
        background: rgba(58, 28, 113, 0.3);
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(215, 109, 119, 0.2);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metrics-container {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: rgba(58, 28, 113, 0.3);
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(215, 109, 119, 0.2);
        flex: 1;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background: linear-gradient(135deg, #D76D77, #5A1C71);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #5A1C71, #D76D77);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(215, 109, 119, 0.2);
    }
    .upload-box {
        border: 2px dashed rgba(215, 109, 119, 0.5);
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        background: rgba(58, 28, 113, 0.2);
        backdrop-filter: blur(10px);
    }
    h1, h2, h3 {
        color: #D76D77 !important;
        font-weight: 600 !important;
    }
    .stMarkdown {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model with caching
@st.cache_resource(show_spinner="Loading the AI model...")
def load_model():
    try:
        model_path = 'brain_tumor_cnn_model.h5'
        # Print current directory contents for debugging
        st.write("Looking for model file...")
        current_dir = os.getcwd()
        files = os.listdir(current_dir)
        st.write(f"Files in current directory: {files}")
        
        if os.path.exists(model_path):
            st.write(f"Found model at: {model_path}")
            model = tf.keras.models.load_model(model_path)
            st.success("Model loaded successfully!")
            return model
        else:
            st.error(f"Model file not found at: {os.path.abspath(model_path)}")
            st.write("Please ensure the model file is uploaded to the repository")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.write("Full error details:", e)
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
