# 🏥 Kidney Tumor Detection System

An end-to-end medical imaging solution that uses Deep Learning to analyze CT scans and detect kidney tumors with high precision. This project features a high-performance **EfficientNetB4** backbone and a modern, interactive web dashboard.

## 📊 Project Highlights
- **Model Accuracy:** ~99.7% on test data.
- **Backbone:** EfficientNetB4 (Transfer Learning).
- **Frontend:** Modern, responsive Medical Dashboard with session statistics and history.
- **Backend:** Flask REST API with optimized image preprocessing.

---

## 📂 Directory Structure

```text
MINOR_P/
├── .venv/                # Virtual environment
├── backend/
│   └── app.py            # Flask Server (Inference API)
├── frontend/
│   └── index.html        # Interactive Web Dashboard
├── model/
│   └── kidney_tumor_model.keras  # Trained EfficientNetB4 weights
├── Results/              # Evaluation metrics
│   ├── confusion_matrix.png
│   └── training_curves.png
├── kaggle_kidney_tumor_train.py  # Original Training Script
├── README.md             # Project Documentation
└── requirements.txt      # Python dependencies