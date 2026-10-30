
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent

HISTOLOGY_DIR = ROOT / "dataset"
OOD_DIR = ROOT / "ood_data" / "out_of_dataset"

MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "ood_detector.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 8
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# =====================================================
# COLLECT LABELED IMAGE PATHS
# =====================================================

def find_images(folder):
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]


histology_paths = []
for class_folder in HISTOLOGY_DIR.iterdir():
    if class_folder.is_dir():
        histology_paths.extend(find_images(class_folder))

ood_paths = find_images(OOD_DIR)

print("Histopathology images:", len(histology_paths))
print("Out-of-dataset images:", len(ood_paths))

if not histology_paths:
    raise ValueError(
        f"No histopathology images found under {HISTOLOGY_DIR}. "
        "Check your dataset folder."
    )

if not ood_paths:
    raise ValueError(
        f"No unrelated images found under {OOD_DIR}. "
        "Add screenshots and other non-histopathology images first."
    )

if len(histology_paths) < 10 or len(ood_paths) < 10:
    raise ValueError(
        "Please collect at least 10 images per category to start. "
        "More varied examples are strongly recommended."
    )

# Label: 0 = histopathology, 1 = out-of-dataset
all_paths = np.array(
    [str(p) for p in histology_paths + ood_paths]
)
all_labels = np.array(
    [0] * len(histology_paths) + [1] * len(ood_paths),
    dtype=np.int32
)

train_paths, val_paths, train_labels, val_labels = (
    train_test_split(
        all_paths,
        all_labels,
        test_size=0.20,
        random_state=SEED,
        stratify=all_labels
    )
)

print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))


# =====================================================
# LOAD AND PREPROCESS IMAGES
# =====================================================

def load_image(path, label):
    contents = tf.io.read_file(path)

    image = tf.io.decode_image(
        contents,
        channels=3,
        expand_animations=False
    )
    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image, tf.cast(label, tf.float32)


def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=SEED
        )

    ds = ds.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(BATCH_SIZE)
    return ds.prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(
    train_paths, train_labels, training=True
)

val_ds = make_dataset(
    val_paths, val_labels
)


# =====================================================
# BUILD THE OUT-OF-DATASET DETECTOR
# =====================================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Initially train only the new classification head
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.30)(x)
outputs = tf.keras.layers.Dense(
    1, activation="sigmoid"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
]

# =====================================================
# TRAIN
# =====================================================

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save the final best model
model.save(str(MODEL_PATH))

print("\nDetector saved to:", MODEL_PATH)


# =====================================================
# BASIC VALIDATION REPORT
# =====================================================

best_model = tf.keras.models.load_model(MODEL_PATH)

probabilities = best_model.predict(val_ds, verbose=1).ravel()

# Initial threshold only. Tune it using validation results.
predicted_labels = (probabilities >= 0.5).astype(int)

print("\nConfusion Matrix")
print("Label 0 = histopathology")
print("Label 1 = out-of-dataset")
print(confusion_matrix(val_labels, predicted_labels))

print("\nClassification Report")
print(
    classification_report(
        val_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=[
            "Histopathology",
            "Out-of-Dataset"
        ],
        zero_division=0
    )
)
