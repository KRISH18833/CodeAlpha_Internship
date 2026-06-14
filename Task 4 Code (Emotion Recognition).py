# =============================================================================
# EMOTION RECOGNITION FROM SPEECH — CodeAlpha Internship Task 2
# Dataset: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
# Objective: Recognize human emotions from speech audio using CNN + LSTM
# =============================================================================

# =============================================================================
# STEP 1: INSTALL REQUIRED LIBRARIES
# Run in terminal before running this script:
#   pip install librosa tensorflow scikit-learn matplotlib seaborn numpy
# =============================================================================

import os
import glob
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import librosa
import librosa.display

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Dense,
    Dropout, BatchNormalization, Flatten
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("TensorFlow Version :", tf.__version__)
print("Librosa Version    :", librosa.__version__)

# =============================================================================
# STEP 2: EMOTION LABELS FROM RAVDESS FILENAME FORMAT
# =============================================================================
# RAVDESS filename format:
# 03-01-06-01-02-01-12.wav
# Position 3 (index 2) = Emotion code
# Emotions: 01=neutral, 02=calm, 03=happy, 04=sad,
#           05=angry, 06=fearful, 07=disgust, 08=surprised
# =============================================================================

EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# We'll use 6 core emotions (removing calm & surprised for cleaner model)
EMOTIONS_USED = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust']

# =============================================================================
# STEP 3: FEATURE EXTRACTION FUNCTION
# =============================================================================

def extract_features(file_path, max_pad_len=174):
    """
    Extract audio features from a .wav file:
    - MFCCs (Mel-Frequency Cepstral Coefficients) — most important for speech
    - Mel Spectrogram                             — frequency over time
    - Chroma Features                             — pitch class energy
    - Zero Crossing Rate                          — how often signal crosses zero
    - RMS Energy                                  — loudness/intensity
    """
    try:
        # Load audio file (sr=22050 is standard sample rate)
        audio, sample_rate = librosa.load(file_path, sr=22050, duration=3.0)

        # --- 1. MFCCs (40 coefficients) ---
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfcc_delta = librosa.feature.delta(mfcc)       # velocity of MFCCs
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)  # acceleration

        # --- 2. Mel Spectrogram ---
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=40)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # --- 3. Chroma Features ---
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, n_chroma=12)

        # --- 4. Zero Crossing Rate ---
        zcr = librosa.feature.zero_crossing_rate(y=audio)

        # --- 5. RMS Energy ---
        rms = librosa.feature.rms(y=audio)

        # Stack all features: (features, time_steps)
        features = np.vstack([mfcc, mfcc_delta, mfcc_delta2, mel_db, chroma, zcr, rms])

        # Pad or truncate to fixed length
        if features.shape[1] < max_pad_len:
            pad_width = max_pad_len - features.shape[1]
            features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
        else:
            features = features[:, :max_pad_len]

        return features.T  # Return shape: (time_steps, features)

    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
        return None

# =============================================================================
# STEP 4: LOAD DATASET AND EXTRACT FEATURES
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Loading RAVDESS Dataset & Extracting Features")
print("=" * 60)


DATASET_PATH = r'D:\Code Alpha Internship\Task 4\audio_speech_actors_01-24'


# Find all .wav files
audio_files = glob.glob(os.path.join(DATASET_PATH, 'Actor_*', '*.wav'))

if len(audio_files) == 0:
    print(f"\n❌ No audio files found in '{DATASET_PATH}'")
    print("   Please check your DATASET_PATH variable and folder structure.")
    print("   Expected structure:")
    print("   RAVDESS/")
    print("   ├── Actor_01/")
    print("   │   ├── 03-01-01-01-01-01-01.wav")
    print("   │   └── ...")
    print("   ├── Actor_02/")
    print("   └── ...")
    exit()

print(f"Found {len(audio_files)} audio files")

X, y = [], []
skipped = 0

for i, file_path in enumerate(audio_files):
    # Extract emotion from filename
    filename   = os.path.basename(file_path)
    parts      = filename.split('-')
    emotion_id = parts[2]
    emotion    = EMOTION_MAP.get(emotion_id, None)

    # Skip emotions we're not using
    if emotion not in EMOTIONS_USED:
        skipped += 1
        continue

    # Extract features
    features = extract_features(file_path)
    if features is not None:
        X.append(features)
        y.append(emotion)

    # Progress update every 100 files
    if (i + 1) % 100 == 0:
        print(f"  Processed {i + 1}/{len(audio_files)} files...")

print(f"\n✅ Feature extraction complete!")
print(f"   Total samples  : {len(X)}")
print(f"   Skipped        : {skipped}")
print(f"   Feature shape  : {np.array(X).shape}")

# Emotion distribution
from collections import Counter
emotion_counts = Counter(y)
print(f"\nEmotion Distribution:")
for emotion, count in sorted(emotion_counts.items()):
    print(f"  {emotion:<12}: {count} samples")

# =============================================================================
# STEP 5: PREPARE DATA
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Preparing Data for Training")
print("=" * 60)

X = np.array(X)   # Shape: (samples, time_steps, features)
y = np.array(y)

# Encode labels: neutral→0, happy→1, sad→2, etc.
label_encoder = LabelEncoder()
y_encoded     = label_encoder.fit_transform(y)
y_cat         = to_categorical(y_encoded)
num_classes   = len(label_encoder.classes_)

print(f"Input shape     : {X.shape}")
print(f"Number of classes: {num_classes}")
print(f"Classes         : {list(label_encoder.classes_)}")

# Normalize features
X_reshaped = X.reshape(-1, X.shape[-1])
scaler     = StandardScaler()
X_scaled   = scaler.fit_transform(X_reshaped).reshape(X.shape)

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_cat, test_size=0.2,
    random_state=42, stratify=y_encoded
)

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")

# =============================================================================
# STEP 6: BUILD CNN + LSTM MODEL
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: Building CNN + LSTM Model")
print("=" * 60)

model = Sequential([
    # --- CNN Block: Extract local patterns from audio features ---
    Conv1D(64, kernel_size=5, activation='relu', padding='same',
           input_shape=(X_train.shape[1], X_train.shape[2])),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(128, kernel_size=5, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    # --- LSTM Block: Learn temporal patterns over time ---
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),

    # --- Fully Connected Layers ---
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\n📐 Architecture Explained:")
print("  Conv1D(64)  → Detect local audio patterns (edges in time)")
print("  Conv1D(128) → Detect complex audio patterns")
print("  LSTM(128)   → Learn long-term speech dependencies")
print("  LSTM(64)    → Refine temporal understanding")
print("  Dense(64)   → High-level emotion features")
print(f"  Dense({num_classes})    → Output: probability for each emotion")

# =============================================================================
# STEP 7: TRAIN THE MODEL
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: Training the Model")
print("=" * 60)

early_stop = EarlyStopping(
    monitor='val_loss', patience=10,
    restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5,
    patience=5, min_lr=1e-6, verbose=1
)

print(f"Training on {X_train.shape[0]} samples...\n")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# =============================================================================
# STEP 8: EVALUATE THE MODEL
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8: Evaluating the Model")
print("=" * 60)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy : {test_acc * 100:.2f}%")
print(f"✅ Test Loss     : {test_loss:.4f}")

y_pred_proba = model.predict(X_test, verbose=0)
y_pred       = np.argmax(y_pred_proba, axis=1)
y_true       = np.argmax(y_test, axis=1)

print("\n📊 Classification Report:")
print(classification_report(
    y_true, y_pred,
    target_names=label_encoder.classes_
))

# =============================================================================
# STEP 9: VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 9: Generating Visualizations")
print("=" * 60)

fig = plt.figure(figsize=(20, 16))
fig.suptitle('Emotion Recognition from Speech — Report', fontsize=15, fontweight='bold')

# --- Plot 1: Emotion Distribution ---
ax1 = fig.add_subplot(2, 3, 1)
emotions_list = list(emotion_counts.keys())
counts_list   = list(emotion_counts.values())
colors = plt.cm.Set2(np.linspace(0, 1, len(emotions_list)))
bars  = ax1.bar(emotions_list, counts_list, color=colors, edgecolor='black')
ax1.bar_label(bars, fmt='%d', padding=3)
ax1.set_title('Emotion Distribution in Dataset', fontweight='bold')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=30)

# --- Plot 2: Training Accuracy ---
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(history.history['accuracy'],     label='Train Accuracy', color='#3498db', lw=2)
ax2.plot(history.history['val_accuracy'], label='Val Accuracy',   color='#e74c3c', lw=2)
ax2.set_title('Model Accuracy over Epochs', fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- Plot 3: Training Loss ---
ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(history.history['loss'],     label='Train Loss', color='#3498db', lw=2)
ax3.plot(history.history['val_loss'], label='Val Loss',   color='#e74c3c', lw=2)
ax3.set_title('Model Loss over Epochs', fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.legend()
ax3.grid(True, alpha=0.3)

# --- Plot 4: Confusion Matrix ---
ax4 = fig.add_subplot(2, 3, 4)
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            linewidths=0.5)
ax4.set_title('Confusion Matrix', fontweight='bold')
ax4.set_xlabel('Predicted Emotion')
ax4.set_ylabel('True Emotion')
ax4.tick_params(axis='x', rotation=30)
ax4.tick_params(axis='y', rotation=0)

# --- Plot 5: Per-Emotion Accuracy ---
ax5 = fig.add_subplot(2, 3, 5)
per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
bars = ax5.bar(label_encoder.classes_, per_class_acc,
               color=colors[:len(label_encoder.classes_)], edgecolor='black')
ax5.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)
ax5.set_title('Per-Emotion Accuracy (%)', fontweight='bold')
ax5.set_ylabel('Accuracy (%)')
ax5.set_ylim(0, 120)
ax5.tick_params(axis='x', rotation=30)

# --- Plot 6: Sample MFCC Visualization ---
ax6 = fig.add_subplot(2, 3, 6)
sample_audio, sr = librosa.load(audio_files[0], sr=22050, duration=3.0)
mfcc_sample = librosa.feature.mfcc(y=sample_audio, sr=sr, n_mfcc=40)
img = librosa.display.specshow(mfcc_sample, x_axis='time', ax=ax6, sr=sr)
ax6.set_title('Sample MFCC Feature Map\n(Input visualization)', fontweight='bold')
ax6.set_ylabel('MFCC Coefficients')
plt.colorbar(img, ax=ax6)

plt.tight_layout()
plt.savefig('emotion_recognition_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Results plot saved as 'emotion_recognition_results.png'")

# =============================================================================
# STEP 10: FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("STEP 10: FINAL SUMMARY")
print("=" * 60)

total_params = model.count_params()
correct      = np.sum(y_pred == y_true)

print(f"\n🏆 Final Test Accuracy : {test_acc * 100:.2f}%")
print(f"   Test Loss           : {test_loss:.4f}")
print(f"   Correct Predictions : {correct} / {len(y_true)}")
print(f"   Wrong Predictions   : {len(y_true) - correct} / {len(y_true)}")
print(f"   Total Parameters    : {total_params:,}")
print(f"   Epochs Trained      : {len(history.history['accuracy'])}")

print("\n📊 Per-Emotion Accuracy:")
for emotion, acc in zip(label_encoder.classes_, per_class_acc):
    print(f"  {emotion:<12}: {acc:.2f}%")

print("\n✅ Emotion Recognition Model Complete!")
print("   Files generated:")
print("   → emotion_recognition_results.png")