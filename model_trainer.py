"""
=============================================================================
Personal Health — Form Quality Classifier Trainer
=============================================================================
Trains a lightweight neural network (MLP) to classify biomechanical form
quality from pose feature vectors. Exports to TensorFlow Lite for Android
deployment.

Pipeline:
  1. Load dataset/training_data.csv
  2. Preprocess (normalize, encode labels)
  3. Train MLP classifier ( 23 → 64 → 32 → 4 classes )
  4. Evaluate on test split
  5. Export as .h5 (Keras) + .tflite (Android)
  6. Save label encoder and normalization stats

Usage:
  pip install tensorflow scikit-learn pandas numpy
  python model_trainer.py
=============================================================================
"""

import json
import os
import sys
import csv
from pathlib import Path

# Graceful imports
try:
    import numpy as np
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("[WARNING] pandas/numpy not installed. Run: pip install pandas numpy")

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARNING] TensorFlow not installed. Run: pip install tensorflow")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.ensemble import RandomForestClassifier  # fallback
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARNING] scikit-learn not installed. Run: pip install scikit-learn")

BASE_DIR = Path(os.path.dirname(__file__))
DATASET_PATH = BASE_DIR / "dataset" / "training_data.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Feature columns (23 input features)
FEATURE_COLS = [
    "hip_angle_l", "hip_angle_r", "knee_angle_l", "knee_angle_r",
    "shoulder_angle_l", "shoulder_angle_r", "elbow_angle_l", "elbow_angle_r",
    "ankle_dorsiflexion_l", "ankle_dorsiflexion_r",
    "trunk_lean", "spine_deviation", "shoulder_hip_sep", "head_forward_pos",
    "com_height_norm", "estimated_jump_height", "limb_symmetry_idx"
]

TARGET_COL = "quality_label"
QUALITY_CLASSES = ["poor", "average", "good", "elite"]  # 0,1,2,3
SPORT_CLASSES = ["vertical_jump", "snatch", "sprint", "javelin", "cricket_bat"]


def load_data():
    """Load and preprocess the training CSV."""
    if not DATASET_PATH.exists():
        print(f"[ERROR] Dataset not found. Run: python generate_dataset.py")
        return None, None, None

    rows = []
    with open(DATASET_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    X = []
    y = []
    sports = []

    for row in rows:
        try:
            features = [float(row[col]) for col in FEATURE_COLS]
            label = row[TARGET_COL]
            sport = row["sport"]
            if label in QUALITY_CLASSES:
                X.append(features)
                y.append(QUALITY_CLASSES.index(label))
                sports.append(SPORT_CLASSES.index(sport) if sport in SPORT_CLASSES else 0)
        except (ValueError, KeyError):
            continue

    print(f"[DATA] Loaded {len(X)} samples")
    if not X:
        return None, None, None

    # Add sport as feature
    X_with_sport = [[*feat, sports[i] / 4.0] for i, feat in enumerate(X)]

    return X_with_sport, y, rows


def train_model(X, y):
    """Train the classifier. Uses TF/Keras if available, sklearn RandomForest as fallback."""

    # Convert to numpy
    X_arr = [[float(v) for v in row] for row in X]
    y_arr = y

    # Normalize features
    mean_vals = [sum(row[i] for row in X_arr) / len(X_arr) for i in range(len(X_arr[0]))]
    std_vals = [max(1e-6, (sum((row[i] - mean_vals[i])**2 for row in X_arr) / len(X_arr))**0.5)
                for i in range(len(X_arr[0]))]

    X_norm = [[(v - mean_vals[i]) / std_vals[i] for i, v in enumerate(row)] for row in X_arr]

    # Save normalization params
    norm_params = {"mean": mean_vals, "std": std_vals, "features": FEATURE_COLS + ["sport_idx"]}
    with open(MODEL_DIR / "norm_params.json", "w") as f:
        json.dump(norm_params, f, indent=2)
    print(f"[NORM] Normalization params saved")

    # Train/test split (80/20)
    n = len(X_norm)
    test_size = int(n * 0.2)
    train_size = n - test_size

    # Shuffle
    import random
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)

    X_train = [X_norm[i] for i in indices[:train_size]]
    X_test = [X_norm[i] for i in indices[test_size:]]
    y_train = [y_arr[i] for i in indices[:train_size]]
    y_test = [y_arr[i] for i in indices[test_size:]]

    print(f"[SPLIT] Train: {len(X_train)}, Test: {len(X_test)}")

    if TF_AVAILABLE:
        return _train_keras(X_train, y_train, X_test, y_test)
    elif SKLEARN_AVAILABLE:
        return _train_sklearn_fallback(X_train, y_train, X_test, y_test)
    else:
        print("[ERROR] Neither TensorFlow nor scikit-learn available.")
        return None


def _train_keras(X_train, y_train, X_test, y_test):
    """Train MLP with Keras and export to TFLite."""
    import numpy as np

    X_tr = np.array(X_train, dtype=np.float32)
    X_te = np.array(X_test, dtype=np.float32)
    y_tr = np.array(y_train, dtype=np.int32)
    y_te = np.array(y_test, dtype=np.int32)

    # One-hot encode
    y_tr_oh = tf.keras.utils.to_categorical(y_tr, num_classes=4)
    y_te_oh = tf.keras.utils.to_categorical(y_te, num_classes=4)

    # Model Architecture: 23+1 → 128 → 64 → 32 → 4
    model = keras.Sequential([
        keras.layers.Input(shape=(X_tr.shape[1],)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(4, activation='softmax')
    ], name="pose_quality_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5)
    ]

    print("[TRAIN] Training Keras MLP...")
    history = model.fit(
        X_tr, y_tr_oh,
        epochs=120,
        batch_size=32,
        validation_data=(X_te, y_te_oh),
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    loss, acc = model.evaluate(X_te, y_te_oh, verbose=0)
    print(f"\n[EVAL] Test Accuracy: {acc*100:.1f}% | Test Loss: {loss:.4f}")

    # Save training metrics
    final_epoch = len(history.history['accuracy'])
    metrics = {
        "final_train_acc": round(history.history['accuracy'][-1], 3),
        "final_val_acc": round(history.history['val_accuracy'][-1], 3),
        "test_accuracy": round(float(acc), 3),
        "test_loss": round(float(loss), 4),
        "epochs_trained": final_epoch,
        "model_type": "MLP_Keras",
        "input_features": len(X_tr[0]),
        "num_classes": 4,
        "class_labels": QUALITY_CLASSES
    }

    # Classification report
    y_pred = np.argmax(model.predict(X_te), axis=1)
    report_text = ""
    for i, cls in enumerate(QUALITY_CLASSES):
        true_pos = sum(1 for t, p in zip(y_te, y_pred) if t == i and p == i)
        total_true = sum(1 for t in y_te if t == i)
        total_pred = sum(1 for p in y_pred if p == i)
        precision = true_pos / max(total_pred, 1)
        recall = true_pos / max(total_true, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        report_text += f"  {cls:8s}: P={precision:.2f} R={recall:.2f} F1={f1:.2f}\n"
    print(f"[REPORT]\n{report_text}")
    metrics["classification_report"] = report_text

    with open(MODEL_DIR / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save Keras model
    keras_path = MODEL_DIR / "pose_classifier.h5"
    model.save(keras_path)
    print(f"[SAVE] Keras model: {keras_path}")

    # Export to TFLite for Android
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # INT8 quantization
        tflite_model = converter.convert()

        tflite_path = MODEL_DIR / "pose_classifier.tflite"
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"[EXPORT] TFLite model: {tflite_path} ({os.path.getsize(tflite_path)//1024}KB)")
    except Exception as e:
        print(f"[WARNING] TFLite export failed: {e}")

    return model


def _train_sklearn_fallback(X_train, y_train, X_test, y_test):
    """Fallback: train a RandomForest when TensorFlow is unavailable."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import pickle

    print("[TRAIN] TensorFlow not available. Training RandomForest fallback...")
    
    clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[EVAL] RandomForest Test Accuracy: {acc*100:.1f}%")

    # Feature importance
    importances = clf.feature_importances_
    feature_names = FEATURE_COLS + ["sport_idx"]
    importance_dict = sorted(zip(feature_names, importances), key=lambda x: -x[1])

    print("[FEATURES] Top 5 most important:")
    for name, imp in importance_dict[:5]:
        print(f"  {name}: {imp:.3f}")

    metrics = {
        "model_type": "RandomForest",
        "test_accuracy": round(float(acc), 3),
        "n_estimators": 200,
        "feature_importances": {n: round(float(v), 4) for n, v in importance_dict}
    }
    with open(MODEL_DIR / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    model_path = MODEL_DIR / "pose_classifier_rf.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"[SAVE] RandomForest model: {model_path}")

    return clf


if __name__ == "__main__":
    print("=" * 60)
    print("ActiveBharat — Sports Pose Classifier Training")
    print("=" * 60)

    X, y, rows = load_data()
    if X is None:
        sys.exit(1)

    label_dist = {cls: y.count(i) for i, cls in enumerate(QUALITY_CLASSES)}
    print(f"[DATA] Label distribution: {label_dist}")

    model = train_model(X, y)
    if model:
        print("\n[DONE] Training complete! Models saved to ml/models/")
        print("  → pose_classifier.h5    (Keras, Python inference)")
        print("  → pose_classifier.tflite (Android deployment)")
        print("  → norm_params.json       (normalization stats)")
        print("\nNext: python api_server.py  (start REST API)")
