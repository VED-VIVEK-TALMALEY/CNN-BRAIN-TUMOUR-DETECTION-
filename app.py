import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import io
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = None
model_path = 'brain_tumor_cnn_model.h5'

def load_model_if_needed():
    global model
    if model is None:
        try:
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                logger.info("Model loaded successfully!")
            else:
                logger.warning(f"Model file not found: {model_path}")
                logger.info("Please run 'python train_model.py' to train and create the model first")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.info("Please ensure the model file is valid or retrain the model")
    return model is not None

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        img = Image.open(image_path)
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
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise ValueError(f"Failed to process image: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if not load_model_if_needed():
        return jsonify({
            'error': 'Model not loaded. Please run "python train_model.py" to train the model first.',
            'details': 'The brain tumor detection model has not been trained yet.'
        })
    
    try:
        # Validate file type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(allowed_extensions)}'})
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Preprocess image
            processed_image = preprocess_image(filepath)
            
            # Make prediction
            prediction = model.predict(processed_image, verbose=0)
            probability = float(prediction[0][0])
            
            # Determine result
            if probability > 0.5:
                result = "Tumor Detected"
                confidence = probability
            else:
                result = "No Tumor"
                confidence = 1 - probability
            
            return jsonify({
                'result': result,
                'confidence': f"{confidence:.2%}",
                'probability': f"{probability:.4f}"
            })
            
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
        
    except ValueError as e:
        return jsonify({'error': str(e)})
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': f'Error processing image: {str(e)}'})

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None,
        'model_path': model_path,
        'model_exists': os.path.exists(model_path) if model_path else False
    })

@app.route('/model-status')
def model_status():
    """Detailed model status endpoint"""
    status = {
        'model_loaded': model is not None,
        'model_path': model_path,
        'model_exists': os.path.exists(model_path) if model_path else False,
        'model_info': None
    }
    
    if model is not None:
        try:
            status['model_info'] = {
                'input_shape': model.input_shape,
                'output_shape': model.output_shape,
                'layers_count': len(model.layers),
                'total_params': model.count_params()
            }
        except Exception as e:
            status['model_info'] = {'error': str(e)}
    
    return jsonify(status)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
