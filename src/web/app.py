"""
Main Flask Application for Brain Tumor Detection
"""

import os
import sys
import logging
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import WEB_CONFIG, MODEL_FILE, UPLOADS_DIR, get_web_paths
from src.utils.image_processor import preprocess_image, validate_image_format, get_image_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Configure app
    app.config['SECRET_KEY'] = WEB_CONFIG['SECRET_KEY']
    app.config['UPLOAD_FOLDER'] = WEB_CONFIG['UPLOAD_FOLDER']
    app.config['MAX_CONTENT_LENGTH'] = WEB_CONFIG['MAX_CONTENT_LENGTH']
    
    # Create uploads directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Load the trained model
    model = None
    try:
        if os.path.exists(MODEL_FILE):
            model = tf.keras.models.load_model(MODEL_FILE)
            logger.info(f"Model loaded successfully from {MODEL_FILE}")
        else:
            logger.warning(f"Model file not found: {MODEL_FILE}")
            logger.info("Please run the training script first")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
    
    # Store model in app context
    app.model = model
    
    # Register routes
    register_routes(app)
    
    return app

def register_routes(app):
    """Register all application routes"""
    
    @app.route('/')
    def index():
        """Main page"""
        web_paths = get_web_paths()
        return render_template('index.html', **web_paths)
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """Handle image prediction requests"""
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if app.model is None:
            return jsonify({
                'error': 'Model not loaded. Please run the training script first.',
                'details': 'The brain tumor detection model has not been trained yet.'
            })
        
        try:
            # Validate file type
            if not validate_image_format(file.filename):
                supported_formats = ', '.join(['.jpg', '.jpeg', '.png', '.bmp'])
                return jsonify({'error': f'Invalid file type. Supported formats: {supported_formats}'})
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Get image info
                image_info = get_image_info(filepath)
                
                # Preprocess image
                processed_image = preprocess_image(filepath)
                
                # Make prediction
                prediction = app.model.predict(processed_image, verbose=0)
                probability = float(prediction[0][0])
                
                # Determine result
                if probability > 0.5:
                    result = "Tumor Detected"
                    confidence = probability
                    risk_level = "High" if probability > 0.8 else "Medium"
                else:
                    result = "No Tumor"
                    confidence = 1 - probability
                    risk_level = "Low"
                
                return jsonify({
                    'result': result,
                    'confidence': f"{confidence:.2%}",
                    'probability': f"{probability:.4f}",
                    'risk_level': risk_level,
                    'image_info': image_info
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
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy', 
            'model_loaded': app.model is not None,
            'model_path': str(MODEL_FILE),
            'model_exists': os.path.exists(MODEL_FILE)
        })
    
    @app.route('/model-status')
    def model_status():
        """Detailed model status endpoint"""
        status = {
            'model_loaded': app.model is not None,
            'model_path': str(MODEL_FILE),
            'model_exists': os.path.exists(MODEL_FILE),
            'model_info': None
        }
        
        if app.model is not None:
            try:
                status['model_info'] = {
                    'input_shape': app.model.input_shape,
                    'output_shape': app.model.output_shape,
                    'layers_count': len(app.model.layers),
                    'total_params': app.model.count_params()
                }
            except Exception as e:
                status['model_info'] = {'error': str(e)}
        
        return jsonify(status)
    
    @app.route('/upload-test')
    def upload_test():
        """Test page for file uploads"""
        web_paths = get_web_paths()
        return render_template('upload_test.html', **web_paths)

def main():
    """Main function to run the application"""
    app = create_app()
    
    logger.info(f"Starting Flask application on {WEB_CONFIG['HOST']}:{WEB_CONFIG['PORT']}")
    logger.info(f"Debug mode: {WEB_CONFIG['DEBUG']}")
    logger.info(f"Upload folder: {WEB_CONFIG['UPLOAD_FOLDER']}")
    
    app.run(
        host=WEB_CONFIG['HOST'],
        port=WEB_CONFIG['PORT'],
        debug=WEB_CONFIG['DEBUG']
    )

if __name__ == '__main__':
    main()
