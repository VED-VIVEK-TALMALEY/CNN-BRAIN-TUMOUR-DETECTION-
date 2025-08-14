import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Set page config
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #FF69B4;
        color: white;
    }
    .stButton>button:hover {
        background-color: #FF1493;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("Brain Tumor Detection System")
st.write("Upload an MRI scan to detect the presence of a brain tumor")

# Load the model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('brain_tumor_cnn_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an MRI scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI scan', use_container_width=True)
        
        # Preprocess the image
        img = image.resize((64, 64))
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if model is not None:
            # Make prediction
            prediction = model.predict(img_array)
            probability = prediction[0][0]
            
            # Display results
            st.write("### Results:")
            if probability > 0.5:
                st.error(f"Brain Tumor Detected (Confidence: {probability:.2%})")
            else:
                st.success(f"No Brain Tumor Detected (Confidence: {(1-probability):.2%})")
                
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
else:
    st.info("Please upload an MRI scan image to begin analysis")
