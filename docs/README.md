# Brain Tumor Detection using CNN

An advanced AI-powered Convolutional Neural Network (CNN) model for accurate brain tumor detection in MRI scans with a modern web interface featuring glass morphism effects.

## 🏗️ Project Structure

```
CNN-BRAIN-TUMOUR-DETECTION-/
├── src/                          # Source code directory
│   ├── models/                   # Model-related code
│   │   ├── __init__.py
│   │   └── cnn_model.py         # CNN architecture & training
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   └── image_processor.py   # Image processing utilities
│   ├── web/                      # Web application
│   │   ├── __init__.py
│   │   └── app.py               # Main Flask app
│   └── training/                 # Training scripts
│       ├── __init__.py
│       └── train_model.py       # Training script
├── static/                       # Static files
│   ├── css/
│   │   └── style.css            # Glass morphism styles
│   ├── js/
│   │   └── main.js              # JavaScript functionality
│   └── images/                   # Static images
├── templates/                    # HTML templates
│   └── index.html               # Main interface
├── data/                         # Data directory
│   ├── raw/                      # Raw dataset
│   ├── processed/                # Processed dataset
│   └── models/                   # Saved models
├── scripts/                      # Utility scripts
│   ├── setup.py                  # Project setup
│   └── organize_dataset.py       # Dataset organization
├── tests/                        # Test files
│   └── __init__.py
├── docs/                         # Documentation
│   └── README.md                 # This file
├── config.py                     # Main configuration
├── main.py                       # Entry point
└── requirements.txt              # Dependencies
```

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

2. **Run the setup script**
   ```bash
   python scripts/setup.py
   ```

3. **Activate virtual environment**
   ```bash
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

4. **Organize your dataset**
   ```bash
   python main.py organize-dataset
   ```

5. **Train the model**
   ```bash
   python main.py train
   ```

6. **Run the web application**
   ```bash
   python main.py web
   ```

7. **Open your browser**
   Navigate to `http://localhost:5000`

## 🎯 Usage

### Main Commands

The project provides a unified command interface through `main.py`:

```bash
# Run web application
python main.py web

# Train the model
python main.py train

# Organize dataset
python main.py organize-dataset

# Setup project
python main.py setup

# Validate dataset
python main.py validate

# Run tests
python main.py test
```

### Training the Model

1. **Organize your dataset** in the following structure:
   ```
   data/processed/brain_tumor_dataset/
   ├── train/
   │   ├── yes/          # Images with tumors
   │   └── no/           # Images without tumors
   └── validation/
       ├── yes/          # Validation images with tumors
       └── no/           # Validation images without tumors
   ```

2. **Run training**:
   ```bash
   python main.py train
   ```

3. **Monitor progress** - the script will show:
   - Number of training samples found
   - Training progress with callbacks
   - Final accuracy metrics

### Using the Web Interface

1. **Upload an MRI scan** by dragging and dropping or clicking to browse
2. **Click "Analyze Image"** to process the scan
3. **View results** including:
   - Detection result (Tumor/No Tumor)
   - Confidence level
   - Probability score
   - Risk level assessment
   - Analysis time

## 🔧 Configuration

### Model Parameters

- **Input size**: 150x150 pixels
- **Architecture**: 4 convolutional blocks with batch normalization
- **Activation**: ReLU for hidden layers, Sigmoid for output
- **Optimizer**: Adam with learning rate scheduling
- **Loss**: Binary crossentropy
- **Metrics**: Accuracy, Precision, Recall, AUC

### Training Parameters

- **Batch size**: 32 (configurable)
- **Epochs**: 30 (configurable)
- **Data augmentation**: Rotation, shift, shear, zoom, flip, brightness
- **Callbacks**: Early stopping, learning rate reduction

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

1. **Import errors**
   - Ensure you're in the project root directory
   - Check that virtual environment is activated
   - Verify all dependencies are installed

2. **Model not loaded**
   - Ensure you've run `python main.py train` first
   - Check that `data/models/brain_tumor_cnn_model.h5` exists

3. **Training failures**
   - Ensure dataset is properly organized
   - Check available memory for training
   - Verify image formats are supported

4. **CSS not loading**
   - Verify static folder structure
   - Check Flask static file configuration

### Debug Mode

Run with debug enabled for detailed error messages:
```bash
python main.py web --debug
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
