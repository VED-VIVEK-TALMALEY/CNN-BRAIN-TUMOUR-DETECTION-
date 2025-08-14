import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import time

# Set page config
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with glassmorphism and Apple-like animations
st.markdown("""
<style>
body, .stApp { background: linear-gradient(135deg, #1E1E1E 60%, #2a004f 100%); }
.glass-card {
        background: rgba(30, 30, 30, 0.85);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 24px;
        margin: 24px 0;
        box-shadow: 0 0 32px 0 #f5bc42, 0 0 8px 2px #fff2;
        border: 2px solid #f5bc42;
        animation: fadeInGlow 1s cubic-bezier(.4,0,.2,1);
        max-width: 95vw;
        box-sizing: border-box;
}
.glass-card:hover {
        box-shadow: 0 0 48px 0 #f5bc42, 0 0 16px 4px #fff5;
        border-color: #ff1493;
        transition: box-shadow 0.3s, border-color 0.3s;
}
.scan-container { position: relative; overflow: hidden; border-radius: 15px; }
.scan-line {
        position: absolute; top: 0; left: 0; width: 100%; height: 4px;
        background: linear-gradient(90deg, transparent, #f5bc42, #ff1493, transparent);
        animation: scanLine 2s linear infinite;
        box-shadow: 0 0 8px 2px #f5bc42;
}
@keyframes fadeInGlow {
        0% { opacity: 0; box-shadow: 0 0 0 0 #f5bc42; }
        100% { opacity: 1; box-shadow: 0 0 32px 0 #f5bc42; }
}
@keyframes scanLine {
        0% { transform: translateY(-100%); }
        100% { transform: translateY(100%); }
}
.stButton > button {
        background: linear-gradient(90deg, #f5bc42 60%, #ff1493 100%);
        color: white;
        border-radius: 12px;
        border: 2px solid #f5bc42;
        font-weight: 600;
        font-size: 1.1em;
        box-shadow: 0 0 12px 0 #f5bc42;
        transition: box-shadow 0.3s, border-color 0.3s;
        padding: 0.7em 1.5em;
}
.stButton > button:hover {
        box-shadow: 0 0 24px 0 #ff1493;
        border-color: #ff1493;
        background: linear-gradient(90deg, #ff1493 60%, #f5bc42 100%);
}
.loader {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 60px;
        margin: 16px 0;
}
.anime-loader {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        border: 6px solid #f5bc42;
        border-top: 6px solid #ff1493;
        animation: spinAnime 1s linear infinite;
        box-shadow: 0 0 16px 2px #f5bc42;
}
@keyframes spinAnime {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
}
@media only screen and (max-width: 600px) {
    .glass-card {
        padding: 10px;
        margin: 10px 0;
        font-size: 1em;
        max-width: 100vw;
    }
    h1, h2, h3 {
        font-size: 1.2em !important;
    }
    .scan-container {
        max-width: 95vw;
    }
    .stImage > img {
        width: 90vw !important;
        height: auto !important;
        max-width: 350px !important;
    }
    .stButton > button {
        font-size: 1em !important;
        padding: 0.5em 1em !important;
    }
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div class="glass-card">
    <h1 style='text-align: center;'>🧠 Brain Tumor Detection System</h1>
    <p style='text-align: center;'>Upload an MRI scan to detect the presence of a brain tumor using advanced AI analysis</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('brain_tumor_cnn_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None
model = load_model()

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align:center;color:#f5bc42;">Upload MRI Image</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an MRI scan image...", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.markdown('<div class="glass-card scan-container" style="max-width:400px;margin:auto;">', unsafe_allow_html=True)
            st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)
            st.image(image, caption='Uploaded MRI scan', width=320)
            st.markdown('</div>', unsafe_allow_html=True)
            img = image.convert('RGB')
            img = img.resize((150, 150), Image.Resampling.LANCZOS)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            if model is not None:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="loader"><div class="anime-loader"></div></div>', unsafe_allow_html=True)
                st.markdown("<h3 style='color:#f5bc42;text-align:center;'>🔍 Analyzing...</h3>", unsafe_allow_html=True)
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                steps = ["Initializing neural network...", "Processing image data...", "Analyzing patterns...", "Detecting anomalies...", "Generating results..."]
                for idx, step in enumerate(steps):
                    status_text.markdown(f"<span style='color:#f5bc42;font-weight:600;'>{step}</span>", unsafe_allow_html=True)
                    progress_bar.progress((idx + 1) / len(steps))
                    time.sleep(0.5)
                prediction = model.predict(img_array, verbose=0)
                probability = float(prediction[0][0])
                progress_bar.empty()
                status_text.empty()
                st.markdown("<h3 style='color:#f5bc42;text-align:center;'>📊 Analysis Results</h3>", unsafe_allow_html=True)
                if probability > 0.7:
                    st.markdown(f"<div style='background:rgba(255,0,0,0.15);padding:24px;border-radius:16px;border:2px solid #f5bc42;box-shadow:0 0 24px 0 #f5bc42;'><h3 style='color:#FF4444;text-shadow:0 0 8px #0d09ed ;'>⚠️ Warning Tumor Detected</h3><p style='font-size:1.2em;font-weight:600;'>Confidence: {probability:.1%}</p><p style='font-size:1em;color:#FF4444;'>Please consult a medical professional immediately.</p></div>", unsafe_allow_html=True)
                elif probability <= 0.3:
                    st.markdown(f"<div style='background:rgba(0,255,0,0.12);padding:24px;border-radius:16px;border:2px solid #44FF44;box-shadow:0 0 24px 0 #44FF44;'><h3 style='color:#44FF44;text-shadow:0 0 8px #44FF44;'>✅ No Tumor Detected</h3><p style='font-size:1.2em;font-weight:600;'>Confidence: {1-probability:.1%}</p><p style='font-size:1em;color:#44FF44;'>Regular check-ups recommended.</p></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='background:rgba(255,255,0,0.12);padding:24px;border-radius:16px;border:2px solid #FFFF44;box-shadow:0 0 24px 0 #FFFF44;'><h3 style='color:#FFFF44;text-shadow:0 0 8px #FFFF44;'>⚠️ Uncertain Results</h3><p style='font-size:1.2em;font-weight:600;'>Confidence: {probability:.1%}</p><p style='font-size:1em;color:#FFFF44;'>Further examination recommended.</p></div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.markdown("<div class='glass-card' style='background:rgba(255,0,0,0.1);'><p>Please try uploading a different MRI scan image.</p></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='glass-card' style='text-align:center;padding:40px;max-width:400px;margin:auto;'><h3 style='color:rgba(255,255,255,0.8);'>👆 Upload an MRI scan to begin analysis</h3><p style='color:rgba(255,255,255,0.6);'>Supported formats: JPG, JPEG, PNG</p></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='glass-card'><h3>ℹ️ About the Analysis</h3><p>This system uses advanced deep learning to analyze MRI scans and detect potential brain tumors.</p><ul style='color:rgba(255,255,255,0.8);'><li>Real-time processing</li><li>High-accuracy detection</li><li>Advanced pattern recognition</li><li>Instant visual feedback</li></ul><p style='font-size:0.8em;color:rgba(255,255,255,0.6);'>Note: This tool is for preliminary screening only and should not be used as a substitute for professional medical diagnosis.</p></div>", unsafe_allow_html=True)
