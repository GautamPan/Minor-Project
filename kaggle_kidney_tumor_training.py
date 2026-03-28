"""
KAGGLE NOTEBOOK - KIDNEY TUMOR DETECTION TRAINING
Complete end-to-end training script for Kaggle
- Auto-loads CT Kidney Dataset from Kaggle
- Trains EfficientNetB4 model with Data Augmentation
- Auto-downloads trained model
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import glob
import shutil

# ==================== KAGGLE SETUP ====================
KAGGLE_ENV = os.path.exists('/kaggle')

def find_kaggle_dataset_path():
    if not KAGGLE_ENV: return None
    print("[*] Scanning /kaggle/input for Normal/Tumor folders...")
    for root, dirs, files in os.walk("/kaggle/input"):
        if 'Normal' in dirs and 'Tumor' in dirs:
            print(f"[✓] Found dataset at: {root}")
            return root
    return None

if KAGGLE_ENV:
    DATASET_PATH = find_kaggle_dataset_path()
    if DATASET_PATH is None:
        DATASET_PATH = "/kaggle/input/ct-kidney-dataset-normal-cyst-tumor-stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
    MODEL_BASE_PATH = "/kaggle/working/kidney_tumor_model"
    VISUALIZATION_PATH = "/kaggle/working"
    print("✓ Running on KAGGLE")
else:
    DATASET_PATH = r"C:\Users\gp890\Downloads\archive\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
    MODEL_BASE_PATH = "model/kidney_tumor_best"
    VISUALIZATION_PATH = "model"
    print("✓ Running LOCALLY")

# ==================== CONFIG ====================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 40
EPOCHS_PHASE2 = 20
VALIDATION_SPLIT = 0.2

print("\n" + "="*80)
print("KIDNEY TUMOR DETECTION - TRAINING")
print("="*80)

# ==================== DATA LOADING ====================
def load_data():
    print("\n[*] Loading images from dataset...")
    images = []
    labels = []
    
    # Normal images (class 0)
    normal_dir = os.path.join(DATASET_PATH, "Normal")
    if os.path.exists(normal_dir):
        normal_images = sorted(glob.glob(os.path.join(normal_dir, "*.jpg")))
        print(f"[✓] Found {len(normal_images)} Normal images")
        
        # 🚀 UPGRADE: Removed artificial limits to use full dataset
        for idx, img_path in enumerate(normal_images):
            try:
                img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                # 🚀 UPGRADE: EfficientNet expects 0-255 scale, removed /255.0 double scaling
                img_array = img_to_array(img) 
                images.append(img_array)
                labels.append(0)
            except Exception:
                pass
    
    # Tumor images (class 1)
    tumor_dir = os.path.join(DATASET_PATH, "Tumor")
    if os.path.exists(tumor_dir):
        tumor_images = sorted(glob.glob(os.path.join(tumor_dir, "*.jpg")))
        print(f"[✓] Found {len(tumor_images)} Tumor images")
        
        for idx, img_path in enumerate(tumor_images):
            try:
                img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(1)
            except Exception:
                pass
    
    if len(images) == 0:
        return None, None
        
    return np.array(images), np.array(labels)

X, y = load_data()
if X is None: sys.exit(1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
)

# 🚀 UPGRADE: Calculate Class Weights for imbalanced dataset
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print(f"\n[✓] Applied Class Weights to balance training: {class_weight_dict}")

# ==================== MODEL BUILDING ====================
print("\n" + "="*80)
print("STEP 2: BUILDING MODEL")
print("="*80)

# 🚀 UPGRADE: Native Data Augmentation block to prevent overfitting
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
], name="data_augmentation")

base = keras.applications.EfficientNetB4(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

model = models.Sequential([
    layers.Input((IMG_SIZE, IMG_SIZE, 3)),
    data_augmentation, 
    base,
    layers.GlobalAveragePooling2D(),
    
    layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    
    layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0003)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(1, activation='sigmoid')
])

# ==================== PHASE 1: TRAIN ====================
print("\n" + "="*80)
print("STEP 3: PHASE 1 - TRAINING WITH FROZEN BASE LAYERS")
print("="*80)

base.trainable = False

# 🐛 FIX: Added name='auc' to prevent KeyErrors during Phase 2
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')] 
)

early_stop = EarlyStopping(monitor='val_auc', patience=8, restore_best_weights=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)

history_phase1 = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS_PHASE1,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

best_auc_phase1 = max(history_phase1.history['val_auc'])
print(f"\n[✓] Phase 1 Complete - Best Val AUC: {best_auc_phase1:.4f}")

# ==================== PHASE 2: FINE-TUNE ====================
print("\n" + "="*80)
print("STEP 4: PHASE 2 - FINE-TUNING LAST LAYERS")
print("="*80)

base.trainable = True
for layer in base.layers[:-25]:
    layer.trainable = False

# 🐛 FIX: Explicitly name metric again
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00005),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')] 
)

history_phase2 = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS_PHASE2,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

best_auc_phase2 = max(history_phase2.history['val_auc'])
print(f"\n[✓] Phase 2 Complete - Best Val AUC: {best_auc_phase2:.4f}")

# ==================== EVALUATION ====================
print("\n" + "="*80)
print("STEP 5: EVALUATION")
print("="*80)

y_pred_proba = model.predict(X_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)

# 🐛 FIX: Calculate Specificity cleanly without crashing
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\n{'Metric':<20} {'Value':<15}")
print("-" * 35)
print(f"{'Accuracy':<20} {accuracy:.4f}")
print(f"{'Precision':<20} {precision:.4f}")
print(f"{'Sensitivity':<20} {recall:.4f}")
print(f"{'Specificity':<20} {specificity:.4f}")
print(f"{'F1-Score':<20} {f1:.4f}")
print(f"{'AUC':<20} {auc_score:.4f}")

# ==================== SAVE MODEL & VISUALIZE ====================
os.makedirs(MODEL_BASE_PATH, exist_ok=True)
model.save(MODEL_BASE_PATH)

# Combine histories
combined_history = {}
for key in history_phase1.history:
    combined_history[key] = history_phase1.history[key] + history_phase2.history[key]

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(combined_history['accuracy'], label='Train')
axes[0].plot(combined_history['val_accuracy'], label='Validation')
axes[0].set_title('Model Accuracy')
axes[0].legend()

axes[1].plot(combined_history['loss'], label='Train')
axes[1].plot(combined_history['val_loss'], label='Validation')
axes[1].set_title('Model Loss')
axes[1].legend()
plt.savefig(os.path.join(VISUALIZATION_PATH, "training_curves.png"))
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Tumor'], yticklabels=['Normal', 'Tumor'])
plt.title('Confusion Matrix')
plt.savefig(os.path.join(VISUALIZATION_PATH, "confusion_matrix.png"))
plt.close()

if KAGGLE_ENV:
    output_model_path = "/kaggle/working/kidney_tumor_model"
    if os.path.exists(output_model_path): shutil.rmtree(output_model_path)
    shutil.copytree(MODEL_BASE_PATH, output_model_path)

print("\n" + "="*80)
print("✓✓✓ TRAINING COMPLETE ✓✓✓")
print("="*80 + "\n")