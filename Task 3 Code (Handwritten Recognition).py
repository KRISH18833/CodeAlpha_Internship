# =============================================================================
# HANDWRITTEN CHARACTER RECOGNITION — CodeAlpha Internship Task 3
# Dataset: MNIST (70,000 handwritten digit images)
# Objective: Identify handwritten digits (0–9) using CNN
# =============================================================================

# =============================================================================
# STEP 1: INSTALL REQUIRED LIBRARIES
# Run these in terminal before running the script:
#   pip install tensorflow numpy matplotlib seaborn scikit-learn requests
# =============================================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import gzip
import struct
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("TensorFlow Version:", tf.__version__)

# =============================================================================
# STEP 2: DOWNLOAD & LOAD MNIST DATASET
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: Downloading & Loading MNIST Dataset")
print("=" * 60)

def download_mnist():
    """Download MNIST from GitHub mirror and return numpy arrays."""
    base = 'https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images' : 't10k-images-idx3-ubyte.gz',
        'test_labels' : 't10k-labels-idx1-ubyte.gz',
    }

    def load_images(path):
        with gzip.open(path) as f:
            f.read(16)
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)

    def load_labels(path):
        with gzip.open(path) as f:
            f.read(8)
            return np.frombuffer(f.read(), dtype=np.uint8)

    data = {}
    for name, fname in files.items():
        if not os.path.exists(fname):
            print(f"  Downloading {fname}...")
            r = requests.get(base + fname)
            with open(fname, 'wb') as fp:
                fp.write(r.content)
        else:
            print(f"  Found {fname} locally.")
        if 'images' in name:
            data[name] = load_images(fname)
        else:
            data[name] = load_labels(fname)

    return (data['train_images'], data['train_labels'],
            data['test_images'],  data['test_labels'])

X_train, y_train, X_test, y_test = download_mnist()

print(f"\nTraining images : {X_train.shape}  → {X_train.shape[0]:,} images of 28×28 pixels")
print(f"Testing images  : {X_test.shape}   → {X_test.shape[0]:,} images of 28×28 pixels")
print(f"Classes         : {np.unique(y_train)}  (digits 0–9)")

# =============================================================================
# STEP 3: DATA PREPROCESSING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Data Preprocessing")
print("=" * 60)

# Normalize pixel values from 0-255 to 0-1
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32')  / 255.0

# Reshape to add channel dimension → (samples, 28, 28, 1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1,  28, 28, 1)

# One-hot encode labels → e.g., 3 becomes [0,0,0,1,0,0,0,0,0,0]
y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat  = to_categorical(y_test,  num_classes=10)

print(f"✅ Pixel values normalized: 0–255  →  0.0–1.0")
print(f"✅ Images reshaped        : (28,28) → (28,28,1) for CNN input")
print(f"✅ Labels one-hot encoded : e.g. 3 → [0,0,0,1,0,0,0,0,0,0]")
print(f"\nFinal shapes:")
print(f"  X_train: {X_train.shape}   y_train: {y_train_cat.shape}")
print(f"  X_test : {X_test.shape}    y_test : {y_test_cat.shape}")

# =============================================================================
# STEP 4: VISUALIZE SAMPLE IMAGES
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Visualizing Sample Images")
print("=" * 60)

fig, axes = plt.subplots(3, 10, figsize=(18, 6))
fig.suptitle('MNIST Dataset — Sample Handwritten Digits (0–9)', fontsize=14, fontweight='bold')

for digit in range(10):
    indices = np.where(y_train == digit)[0]
    for row in range(3):
        ax = axes[row, digit]
        ax.imshow(X_train[indices[row]].reshape(28, 28), cmap='gray')
        ax.axis('off')
        if row == 0:
            ax.set_title(str(digit), fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('sample_digits.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Sample digits plot saved as 'sample_digits.png'")

# =============================================================================
# STEP 5: BUILD CNN MODEL
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Building CNN Model")
print("=" * 60)

model = Sequential([
    # --- Block 1: First Convolution ---
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # --- Block 2: Second Convolution ---
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # --- Fully Connected Layers ---
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')   # 10 output classes (digits 0–9)
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\n📐 Model Architecture Explained:")
print("  Conv2D(32)     → Detect basic edges and patterns")
print("  Conv2D(64)     → Detect complex shapes and curves")
print("  MaxPooling2D   → Reduce image size, keep important features")
print("  BatchNorm      → Stabilize and speed up training")
print("  Dropout        → Prevent overfitting")
print("  Dense(256)     → Learn high-level combinations")
print("  Dense(10)      → Output probabilities for digits 0–9")

# =============================================================================
# STEP 6: TRAIN THE MODEL
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: Training the Model")
print("=" * 60)

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss', patience=5,
    restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5,
    patience=3, min_lr=1e-6, verbose=1
)

print("Training CNN on 60,000 images...\n")
history = model.fit(
    X_train, y_train_cat,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# =============================================================================
# STEP 7: EVALUATE THE MODEL
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: Evaluating the Model")
print("=" * 60)

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n✅ Test Accuracy : {test_acc * 100:.2f}%")
print(f"✅ Test Loss     : {test_loss:.4f}")

y_pred_proba = model.predict(X_test, verbose=0)
y_pred       = np.argmax(y_pred_proba, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=[f'Digit {i}' for i in range(10)]))

# =============================================================================
# STEP 8: VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8: Generating Visualizations")
print("=" * 60)

fig = plt.figure(figsize=(20, 16))
fig.suptitle('Handwritten Character Recognition — CNN Report', fontsize=15, fontweight='bold')

# --- Plot 1: Training Accuracy ---
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(history.history['accuracy'],     label='Train Accuracy', color='#3498db', lw=2)
ax1.plot(history.history['val_accuracy'], label='Val Accuracy',   color='#e74c3c', lw=2)
ax1.set_title('Model Accuracy', fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Plot 2: Training Loss ---
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(history.history['loss'],     label='Train Loss', color='#3498db', lw=2)
ax2.plot(history.history['val_loss'], label='Val Loss',   color='#e74c3c', lw=2)
ax2.set_title('Model Loss', fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- Plot 3: Confusion Matrix ---
ax3 = fig.add_subplot(2, 3, 3)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=range(10), yticklabels=range(10),
            linewidths=0.5, annot_kws={"size": 8})
ax3.set_title('Confusion Matrix', fontweight='bold')
ax3.set_xlabel('Predicted Label')
ax3.set_ylabel('True Label')

# --- Plot 4: Per-class Accuracy ---
ax4 = fig.add_subplot(2, 3, 4)
per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
bars = ax4.bar(range(10), per_class_acc, color='steelblue', edgecolor='black')
ax4.bar_label(bars, fmt='%.1f%%', padding=2, fontsize=8)
ax4.set_title('Per-Digit Accuracy (%)', fontweight='bold')
ax4.set_xlabel('Digit')
ax4.set_ylabel('Accuracy (%)')
ax4.set_xticks(range(10))
ax4.set_ylim(0, 115)

# --- Plot 5: Correct Predictions ---
ax5 = fig.add_subplot(2, 3, 5)
correct_idx   = np.where(y_pred == y_test)[0][:10]
incorrect_idx = np.where(y_pred != y_test)[0][:5]
sample_idx    = np.concatenate([correct_idx[:5], incorrect_idx])
grid = [X_test[i].reshape(28, 28) for i in sample_idx]
combined = np.concatenate(grid, axis=1)
ax5.imshow(combined, cmap='gray')
ax5.axis('off')
labels = [f"✓{y_test[i]}" for i in correct_idx[:5]] + \
         [f"✗{y_test[i]}→{y_pred[i]}" for i in incorrect_idx]
ax5.set_title('Predictions: ✓ Correct | ✗ Wrong (True→Pred)',
              fontweight='bold', fontsize=9)

# --- Plot 6: Prediction Confidence ---
ax6 = fig.add_subplot(2, 3, 6)
sample = X_test[incorrect_idx[0]]
probs  = y_pred_proba[incorrect_idx[0]]
bars   = ax6.bar(range(10), probs * 100, color='#e74c3c', edgecolor='black', alpha=0.8)
ax6.bar_label(bars, fmt='%.1f%%', padding=2, fontsize=7)
ax6.set_xticks(range(10))
ax6.set_title(f'Confidence for a Misclassified Sample\n'
              f'(True: {y_test[incorrect_idx[0]]}, Predicted: {y_pred[incorrect_idx[0]]})',
              fontweight='bold', fontsize=9)
ax6.set_xlabel('Digit Class')
ax6.set_ylabel('Confidence (%)')

plt.tight_layout()
plt.savefig('cnn_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ CNN results plot saved as 'cnn_results.png'")

# =============================================================================
# STEP 9: PREDICT ON CUSTOM SAMPLES
# =============================================================================
print("\n" + "=" * 60)
print("STEP 9: Predicting on Random Test Samples")
print("=" * 60)

fig, axes = plt.subplots(2, 10, figsize=(18, 4))
fig.suptitle('Random Predictions — Green=Correct, Red=Wrong', fontsize=13, fontweight='bold')

random_idx = np.random.choice(len(X_test), 10, replace=False)
for col, idx in enumerate(random_idx):
    img   = X_test[idx].reshape(28, 28)
    true  = y_test[idx]
    pred  = y_pred[idx]
    conf  = y_pred_proba[idx][pred] * 100
    color = '#2ecc71' if pred == true else '#e74c3c'

    axes[0, col].imshow(img, cmap='gray')
    axes[0, col].axis('off')
    axes[0, col].set_title(f'T:{true}', fontsize=10)

    axes[1, col].imshow(img, cmap='gray')
    axes[1, col].axis('off')
    axes[1, col].set_title(f'P:{pred}\n{conf:.0f}%', fontsize=9,
                            color='green' if pred == true else 'red')

plt.tight_layout()
plt.savefig('random_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Random predictions saved as 'random_predictions.png'")

# =============================================================================
# STEP 10: FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("STEP 10: FINAL SUMMARY")
print("=" * 60)

total_params = model.count_params()
correct      = np.sum(y_pred == y_test)

print(f"\n🏆 Final Test Accuracy : {test_acc * 100:.2f}%")
print(f"   Test Loss           : {test_loss:.4f}")
print(f"   Correct Predictions : {correct:,} / {len(y_test):,}")
print(f"   Wrong Predictions   : {len(y_test) - correct:,} / {len(y_test):,}")
print(f"   Total Parameters    : {total_params:,}")
print(f"   Epochs Trained      : {len(history.history['accuracy'])}")

print("\n📊 Per-Digit Accuracy:")
for digit in range(10):
    print(f"   Digit {digit}: {per_class_acc[digit]:.2f}%")

print("\n✅ Handwritten Character Recognition Complete!")
print("   Files generated:")
print("   → sample_digits.png")
print("   → cnn_results.png")
print("   → random_predictions.png")