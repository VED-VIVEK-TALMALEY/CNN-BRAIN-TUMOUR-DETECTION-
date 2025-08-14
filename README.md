<<<<<<< HEAD
# Brain Tumor Detection using CNN

An advanced AI-powered Convolutional Neural Network (CNN) model for accurate brain tumor detection in MRI scans with a modern web interface featuring glass morphism effects.

## 🧠 Features

- **Advanced CNN Model**: Deep learning model trained on brain MRI datasets
- **Real-time Analysis**: Instant tumor detection with confidence scores
- **Modern UI**: Glass morphism design with smooth animations
- **Drag & Drop**: Easy image upload interface
- **Multiple Formats**: Supports JPG, PNG, JPEG, and BMP files
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.15.0
- Flask 2.3.3
- Other dependencies (see requirements.txt)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CNN-BRAIN-TUMOUR-DETECTION-
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model**
   ```bash
   python train_model.py
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:5000`

## 📁 Project Structure

```
CNN-BRAIN-TUMOUR-DETECTION-/
├── app.py                 # Main Flask application
├── cnn_model.py          # CNN model architecture
├── train_model.py        # Model training script
├── requirements.txt      # Python dependencies
├── brain_tumor_dataset/  # Training dataset
│   ├── train/
│   │   ├── yes/         # Tumor images
│   │   └── no/          # No tumor images
│   └── validation/
│       ├── yes/         # Validation tumor images
│       └── no/          # Validation no tumor images
├── static/
│   └── css/
│       └── style.css    # Glass morphism styles
├── templates/
│   └── index.html       # Web interface
└── uploads/             # Temporary file storage
```

## 🎯 Usage

### Training the Model

1. **Organize your dataset** in the following structure:
   ```
   brain_tumor_dataset/
   ├── train/
   │   ├── yes/          # Images with tumors
   │   └── no/           # Images without tumors
   └── validation/
       ├── yes/          # Validation images with tumors
       └── no/           # Validation images without tumors
   ```

2. **Run training**:
   ```bash
   python train_model.py
   ```

3. **Monitor progress** - the script will show:
   - Number of training samples found
   - Training progress
   - Final accuracy metrics

### Using the Web Interface

1. **Upload an MRI scan** by dragging and dropping or clicking to browse
2. **Click "Analyze Image"** to process the scan
3. **View results** including:
   - Detection result (Tumor/No Tumor)
   - Confidence level
   - Probability score
   - Analysis time

## 🔧 Configuration

### Model Parameters

- **Input size**: 150x150 pixels
- **Architecture**: 4 convolutional layers + dense layers
- **Activation**: ReLU for hidden layers, Sigmoid for output
- **Optimizer**: Adam
- **Loss**: Binary crossentropy

### Training Parameters

- **Batch size**: 32
- **Epochs**: 30
- **Data augmentation**: Rotation, shift, shear, zoom, flip

## 📊 API Endpoints

- `GET /` - Main web interface
- `POST /predict` - Image analysis endpoint
- `GET /health` - Health check
- `GET /model-status` - Detailed model information

## 🎨 Glass Morphism Design

The interface features modern glass morphism effects:
- **Backdrop blur**: 20px blur for glass-like transparency
- **Subtle borders**: Semi-transparent white borders
- **Layered shadows**: Multiple shadow layers for depth
- **Smooth animations**: CSS transitions and keyframe animations
- **Responsive layout**: Adapts to different screen sizes

## 🐛 Troubleshooting

### Common Issues

1. **Model not loaded**
   - Ensure you've run `python train_model.py` first
   - Check that `brain_tumor_cnn_model.h5` exists

2. **Import errors**
   - Verify all dependencies are installed
   - Check Python version compatibility

3. **Training failures**
   - Ensure dataset is properly organized
   - Check available memory for training

4. **CSS not loading**
   - Verify static folder structure
   - Check Flask static file configuration

### Debug Mode

Run with debug enabled for detailed error messages:
```bash
python app.py
```

## 📈 Performance

- **Training time**: ~30-60 minutes (depending on hardware)
- **Inference time**: ~2-3 seconds per image
- **Model size**: ~50-100 MB
- **Accuracy**: Typically 85-95% on validation data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Brain tumor dataset contributors
- TensorFlow and Keras communities
- Flask web framework
- Modern CSS techniques and glass morphism design

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments
3. Open an issue on GitHub
4. Check the model status endpoint: `/model-status`

---

**Note**: This is a research and educational tool. Always consult medical professionals for actual medical diagnosis. 
=======
# CNN-BRAIN-TUMOUR-DETECTION-
BRAIN TUMOUR DETECTION USING CNN MODEL 


day 1-- 14/08/2025
main project deployement start 
>>>>>>> 2f2b27b2ee32969ac5e68aeec9469d014179769b
