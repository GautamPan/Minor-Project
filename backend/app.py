import os
import numpy as np
from PIL import Image
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras import layers, models

# ==================== 1. DYNAMIC PATH SETUP ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', 'kidney_tumor_model.keras')
STATIC_DIR = os.path.join(PROJECT_ROOT, 'frontend')

# ==================== 2. APP CONFIGURATION ====================
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='')
CORS(app) 

IMG_SIZE = 224
CLASS_NAMES = ['Normal', 'Tumor']
model = None

# ==================== 3. MODEL RECONSTRUCTION ====================

def load_trained_model():
    """Builds the model skeleton and loads weights to avoid layer mismatch errors."""
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"[✗] ERROR: Model file not found at {MODEL_PATH}")
            return False

        print(f"[*] Reconstructing EfficientNetB4 skeleton...")
        
        # Build the exact base used in Kaggle
        base = tf.keras.applications.EfficientNetB4(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights=None # Weights will be loaded from your file
        )
        
        # Rebuild the Sequential stack exactly as per your training script
        skeleton = models.Sequential([
            layers.Input((IMG_SIZE, IMG_SIZE, 3)),
            # Note: We skip data_augmentation here as it's only for training
            base,
            layers.GlobalAveragePooling2D(),
            
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        print(f"[*] Loading trained weights into skeleton...")
        # This is the 'magic' line that fixes the ValueError
        skeleton.load_weights(MODEL_PATH)
        
        model = skeleton
        print("[✓] AI System Online & Ready.")
        return True

    except Exception as e:
        print(f"[✗] CRITICAL ERROR: {e}")
        traceback.print_exc()
        return False

# ==================== 4. INFERENCE LOGIC ====================

def preprocess_image(image_file):
    """Standardizes input for the model."""
    img = Image.open(image_file).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    # EfficientNet expects [0, 255] float values
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==================== 5. API ROUTES ====================

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    
    file = request.files['file']
    
    try:
        # Run Prediction
        img_array = preprocess_image(file)
        prediction_raw = model.predict(img_array, verbose=0)[0][0]
        
        # Binary Classification Logic
        is_tumor = prediction_raw > 0.5
        class_name = "Tumor" if is_tumor else "Normal"
        confidence = float(prediction_raw if is_tumor else 1 - prediction_raw)

        return jsonify({
            'prediction': class_name,
            'confidence': round(confidence, 4),
            'confidence_percent': round(confidence * 100, 2)
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_ready': model is not None}), 200

# ==================== 6. STARTUP ====================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  KIDNEY TUMOR AI - SECURE BACKEND  ")
    print("="*50)
    
    if load_trained_model():
        app.run(host='localhost', port=5000, debug=False)
        
