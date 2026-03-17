<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.15-ff6f00?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Keras-3.x-d00000?logo=keras&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-2.3-black?logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-live-ff4b4b?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-ready-2496ed?logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

# 🧠 Brain Tumour Detection — Deep Learning Web App

**A CNN-based MRI classification system** that detects the presence of brain tumours from MRI scans, served through two interfaces: a Flask web app with glass-morphism UI and a Streamlit demo.

> _"Upload an MRI scan. Get an instant tumour/no-tumour prediction with confidence score."_

🔴 **Live Demo:** [https://braintumoro.streamlit.app/](https://braintumoro.streamlit.app/)

---

## 📊 Results

### Accuracy Improvement Journey

| Run | Model | Data Augmentation | Val Accuracy | Val Loss |
|-----|-------|-------------------|-------------|----------|
| Baseline | `cnn_model.py` (4-layer CNN) | None | ~85% | — |
| Enhanced | `enhanced_cnn_model.py` | Rotation, shift, shear, zoom, flip | **~95%** | ↓ 12% |

> **Best checkpoint saved as** `brain_tumor_cnn_model.h5` — trained over 30 epochs, batch size 32, Adam optimiser, binary crossentropy loss.

### Performance Metrics

| Metric | Value |
|--------|-------|
| Validation Accuracy | ~95% |
| Inference Time | ~2–3 seconds per image |
| Model Size | ~50–100 MB |
| Users Served (Live Demo) | 150+ |
| GitHub Stars | 5+ |

---

## 🏗️ Architecture

```
+========================= BRAIN TUMOUR DETECTION SYSTEM =========================+
|                                                                                 |
|  USER LAYER                                                                     |
|  +--------------------------------------------------------------------------+  |
|  |  Browser (Flask: localhost:5000)  OR  Streamlit (braintumoro.streamlit)  |  |
|  |  - Drag & Drop MRI upload                                                |  |
|  |  - Glass morphism UI (Flask) / Clean demo UI (Streamlit)                 |  |
|  |  - Real-time confidence score + probability display                      |  |
|  +-------------------------------------+------------------------------------+  |
|                                        |                                        |
|                                HTTP POST /predict                               |
|                                        |                                        |
|  BACKEND LAYER                         |                                        |
|  +--------------------------------------------------------------------------+  |
|  |  app.py (Flask)  /  streamlit_app.py (Streamlit)                        |  |
|  |                                                                          |  |
|  |  Endpoints:                                                              |  |
|  |    GET  /              — Web interface                                   |  |
|  |    POST /predict       — Image upload + model inference                  |  |
|  |    GET  /health        — Health check                                    |  |
|  |    GET  /model-status  — Model metadata                                  |  |
|  +-------------------------------------+------------------------------------+  |
|                                        |                                        |
|                               Model inference                                   |
|                                        |                                        |
|  MODEL LAYER (TensorFlow + Keras)      |                                        |
|  +--------------------------------------------------------------------------+  |
|  |                                                                          |  |
|  |   MRI Input Image (150×150 px, grayscale/RGB)                           |  |
|  |          |                                                               |  |
|  |          v                                                               |  |
|  |   [Conv2D + ReLU + MaxPool]  × 4  ← Feature extraction layers           |  |
|  |          |                                                               |  |
|  |          v                                                               |  |
|  |   [Flatten → Dense → Dropout]    ← Classification head                  |  |
|  |          |                                                               |  |
|  |          v                                                               |  |
|  |   [Dense(1) + Sigmoid]            ← Binary output (Tumour / No Tumour)  |  |
|  |          |                                                               |  |
|  |          v                                                               |  |
|  |   Confidence Score + Prediction Label                                   |  |
|  |                                                                          |  |
|  +--------------------------------------------------------------------------+  |
|                                                                                 |
|  DATA LAYER                                                                     |
|  +--------------------------------------------------------------------------+  |
|  |  brain_tumor_dataset/train/yes   — MRI scans with tumour                |  |
|  |  brain_tumor_dataset/train/no    — MRI scans without tumour             |  |
|  |  brain_tumor_dataset/validation/ — Held-out evaluation set              |  |
|  |  brain_tumor_cnn_model.h5        — Saved trained weights                |  |
|  +--------------------------------------------------------------------------+  |
|                                                                                 |
+=================================================================================+
```

### Model Variants

| File | Description |
|------|-------------|
| `cnn_model.py` | Baseline 4-layer CNN — ReLU activations, Sigmoid output |
| `enhanced_cnn_model.py` | Improved variant with stronger augmentation and tuned architecture |
| `train_model.py` | Trains baseline model |
| `train_enhanced_model.py` | Trains enhanced model (achieves ~95% val accuracy) |

---

## 📁 Project Structure

```
CNN-BRAIN-TUMOUR-DETECTION-/
│
├── app.py                        # Flask web application
├── streamlit_app.py              # Streamlit demo interface
├── cnn_model.py                  # Baseline CNN architecture
├── enhanced_cnn_model.py         # Enhanced CNN architecture
├── train_model.py                # Baseline training script
├── train_enhanced_model.py       # Enhanced training script
├── config.py                     # Model and app configuration
├── main.py                       # CLI entry point
├── reorganize_dataset.py         # Dataset preparation utility
├── brain_tumor_cnn_model.h5      # Saved model weights (Keras)
├── requirements.txt              # Flask app dependencies
├── requirements_enhanced.txt     # Enhanced model dependencies
├── requirements_streamlit.txt    # Streamlit app dependencies
├── Dockerfile                    # Docker container config
├── Procfile                      # Deployment process file
├── runtime.txt                   # Python runtime spec
├── ACCURACY_IMPROVEMENT_GUIDE.md # Guide for improving accuracy further
│
├── src/                          # Core source modules
├── scripts/                      # Utility scripts
├── tests/                        # Test suite
├── docs/                         # Documentation
│
├── static/
│   └── css/
│       └── style.css             # Glass morphism UI styles
│
└── templates/
    └── index.html                # Flask web interface
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip
- ~500 MB disk space (model + dataset)

### Option 1 — Streamlit Demo (Fastest)

```bash
git clone https://github.com/VED-VIVEK-TALMALEY/CNN-BRAIN-TUMOUR-DETECTION-.git
cd CNN-BRAIN-TUMOUR-DETECTION-
pip install -r requirements_streamlit.txt
streamlit run streamlit_app.py
```
Or just visit: **[https://braintumoro.streamlit.app/](https://braintumoro.streamlit.app/)**

---

### Option 2 — Flask Web App (Local)

```bash
# 1. Clone
git clone https://github.com/VED-VIVEK-TALMALEY/CNN-BRAIN-TUMOUR-DETECTION-.git
cd CNN-BRAIN-TUMOUR-DETECTION-

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (skip if brain_tumor_cnn_model.h5 already exists)
python train_enhanced_model.py

# 5. Run
python app.py
```

Open **http://localhost:5000** in your browser.

---

### Option 3 — Docker

```bash
docker build -t brain-tumour-detection .
docker run -p 5000:5000 brain-tumour-detection
```

---

## 🎯 Model Configuration

### CNN Architecture

| Layer | Type | Details |
|-------|------|---------|
| Input | — | 150×150 px image |
| 1 | Conv2D + ReLU + MaxPool | Feature extraction |
| 2 | Conv2D + ReLU + MaxPool | Feature extraction |
| 3 | Conv2D + ReLU + MaxPool | Feature extraction |
| 4 | Conv2D + ReLU + MaxPool | Feature extraction |
| 5 | Flatten + Dense + Dropout | Classification head |
| Output | Dense(1) + Sigmoid | Binary prediction |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Input size | 150×150 px |
| Batch size | 32 |
| Epochs | 30 |
| Optimiser | Adam |
| Loss | Binary crossentropy |
| Augmentation | Rotation, shift, shear, zoom, horizontal flip |

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/predict` | POST | Upload MRI image → returns prediction + confidence |
| `/health` | GET | Service health check |
| `/model-status` | GET | Model metadata and load status |

### Example `/predict` Response

```json
{
  "prediction": "Tumor Detected",
  "confidence": 0.947,
  "probability": 0.947,
  "analysis_time": "2.1s"
}
```

---

## 🖥️ Interface

The Flask app features a **glass morphism UI**:
- Backdrop blur + semi-transparent panels
- Drag & drop MRI upload
- Animated confidence score display
- Responsive layout (desktop + mobile)

The Streamlit app provides a clean, minimal demo interface for quick testing.

---

## 🐛 Troubleshooting

| Issue | Fix |
|-------|-----|
| `brain_tumor_cnn_model.h5` not found | Run `python train_enhanced_model.py` first |
| Import errors | Verify dependencies with `pip install -r requirements.txt` |
| Training failures | Check dataset folder structure matches `brain_tumor_dataset/train/yes` and `.../no` |
| CSS not loading | Verify `static/css/style.css` exists and Flask static config is correct |

---

## 📦 Dataset

**Brain MRI Images** — sourced from Kaggle

| Detail | Info |
|--------|------|
| Source | [Kaggle — Brain MRI Images](https://www.kaggle.com/datasets/ashfakyeafi/brain-mri-images) |
| Author | ashfakyeafi |
| Classes | 2 — `yes` (tumour present), `no` (no tumour) |
| Format | JPEG/PNG MRI scans |
| Split | `train/` and `validation/` directories |

### Dataset Setup

```bash
# Option 1 — Kaggle CLI
kaggle datasets download -d ashfakyeafi/brain-mri-images
unzip brain-mri-images.zip -d brain_tumor_dataset/

# Option 2 — Manual
# Download from https://www.kaggle.com/datasets/ashfakyeafi/brain-mri-images
# Extract and organize as:
brain_tumor_dataset/
├── train/
│   ├── yes/    ← MRI scans with tumour
│   └── no/     ← MRI scans without tumour
└── validation/
    ├── yes/
    └── no/
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Submit a pull request

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

> ⚠️ **Medical Disclaimer**: This is a research and educational project. It is not a certified medical device. Always consult qualified medical professionals for diagnosis and treatment decisions.

---

## 🙏 Acknowledgements

- **Dataset**: [Brain MRI Images](https://www.kaggle.com/datasets/ashfakyeafi/brain-mri-images) by ashfakyeafi on Kaggle
- TensorFlow and Keras communities
- Flask and Streamlit open-source teams

---

<p align="center">
  <b>Built with ❤️ for medical AI research</b><br/>
  <i>Brain Tumour Detection — Making AI useful in healthcare</i><br/>
  by <a href="https://github.com/VED-VIVEK-TALMALEY">Ved Vivek Talmaley</a>
</p>
