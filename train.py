import json
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_DIR = (
    r"C:\Gastric_Cancer_AI\dataset"
    r"\HMU-GC-HE-30K\all_image"
)
IMG_SIZE = (300, 300)
BATCH_SIZE = 16
SEED = 123

INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 3

MODEL_DIR = "model"

BEST_MODEL_PHASE1 = os.path.join(
    MODEL_DIR, "efficientnet_b3_phase1_best.keras"
)
BEST_MODEL_PHASE2 = os.path.join(
    MODEL_DIR, "efficientnet_b3_phase2_best.keras"
)
FINAL_MODEL = os.path.join(
    MODEL_DIR, "efficientnet_b3.keras"
)
HISTORY_FILE = os.path.join(
    MODEL_DIR, "training_history.json"
)
CLASS_NAMES_FILE = os.path.join(
    MODEL_DIR, "class_names.json"
)

EXPECTED_CLASSES = [
    "ADI", "DEB", "LYM", "MUC",
    "MUS", "NOR", "STR", "TUM"
]

os.makedirs(MODEL_DIR, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)

# ============================================================
# 2. CHECK DATASET
# ============================================================

if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset folder not found:\n{os.path.abspath(DATASET_DIR)}"
    )

print("\n" + "=" * 55)
print("GASTRIC CANCER AI - MODEL TRAINING")
print("=" * 55)
print("TensorFlow version:", tf.__version__)
print("Dataset:", os.path.abspath(DATASET_DIR))
print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)
print("Initial epochs:", INITIAL_EPOCHS)
print("Fine-tuning epochs:", FINE_TUNE_EPOCHS)
print("=" * 55)

# ============================================================
# 3. LOAD TRAINING DATA
# ============================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True
)

# ============================================================
# 4. LOAD VALIDATION DATA
# ============================================================

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True
)

# ============================================================
# 5. VERIFY CLASS NAMES AND ORDER
# ============================================================

class_names = train_ds.class_names
validation_classes = val_ds.class_names

if class_names != validation_classes:
    raise ValueError(
        "Training and validation class names do not match."
    )

if class_names != EXPECTED_CLASSES:
    raise ValueError(
        f"Dataset class order mismatch!\n"
        f"Expected: {EXPECTED_CLASSES}\n"
        f"Found: {class_names}"
    )

NUM_CLASSES = len(class_names)

print("\nVerified class order:")

for index, name in enumerate(class_names):
    print(f"{index}: {name}")

print("\nNumber of classes:", NUM_CLASSES)

# Save class order for use by prediction code.
with open(CLASS_NAMES_FILE, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=4)

# ============================================================
# 6. DATASET PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ============================================================
# 7. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
    ],
    name="data_augmentation"
)

# ============================================================
# 8. LOAD EFFICIENTNET-B3
# ============================================================

base_model = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(300, 300, 3)
)

# Freeze pretrained layers for Phase 1.
base_model.trainable = False

# ============================================================
# 9. BUILD MODEL
# ============================================================

inputs = layers.Input(
    shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    name="input_image"
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)

x = layers.Dropout(
    0.30,
    name="dropout"
)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="predictions"
)(x)

model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="gastric_cancer_efficientnet_b3"
)

# ============================================================
# 10. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# 11. PHASE 1 CALLBACKS
# ============================================================

checkpoint1 = ModelCheckpoint(
    BEST_MODEL_PHASE1,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping1 = EarlyStopping(
    monitor="val_accuracy",
    patience=2,
    mode="max",
    restore_best_weights=True,
    verbose=1
)

reduce_lr1 = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=1,
    min_lr=1e-6,
    verbose=1
)

# ============================================================
# 12. PHASE 1: TRANSFER LEARNING
# ============================================================

print("\n" + "=" * 55)
print("PHASE 1: TRANSFER LEARNING")
print("=" * 55)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=[
        checkpoint1,
        early_stopping1,
        reduce_lr1
    ]
)

if not os.path.exists(BEST_MODEL_PHASE1):
    raise FileNotFoundError(
        "Phase 1 did not create a best-model checkpoint."
    )

best_acc1 = max(
    history1.history["val_accuracy"]
)

print(f"\nBest Phase 1 validation accuracy: {best_acc1:.4f}")

# ============================================================
# 13. PHASE 2: FINE-TUNING
# ============================================================

print("\n" + "=" * 55)
print("PHASE 2: FINE-TUNING")
print("=" * 55)

# Load the best Phase 1 model before fine-tuning.
model = tf.keras.models.load_model(
    BEST_MODEL_PHASE1,
    compile=False
)

# Get the pretrained EfficientNet base model.
base_model = model.get_layer("efficientnetb3")
base_model.trainable = True

# Freeze most EfficientNet layers.
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen.
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

# Recompile with a smaller learning rate.
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint2 = ModelCheckpoint(
    BEST_MODEL_PHASE2,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping2 = EarlyStopping(
    monitor="val_accuracy",
    patience=2,
    mode="max",
    restore_best_weights=True,
    verbose=1
)

reduce_lr2 = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=1,
    min_lr=1e-7,
    verbose=1
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=len(history1.history["loss"]),
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    callbacks=[
        checkpoint2,
        early_stopping2,
        reduce_lr2
    ]
)

if not os.path.exists(BEST_MODEL_PHASE2):
    raise FileNotFoundError(
        "Phase 2 did not create a best-model checkpoint."
    )

best_acc2 = max(
    history2.history["val_accuracy"]
)

print(f"\nBest Phase 2 validation accuracy: {best_acc2:.4f}")

# ============================================================
# 14. SELECT THE BEST CHECKPOINT
# ============================================================

print("\n" + "=" * 55)
print("COMPARING BEST VALIDATION RESULTS")
print("=" * 55)

print(f"Phase 1 accuracy: {best_acc1:.4f}")
print(f"Phase 2 accuracy: {best_acc2:.4f}")

if best_acc2 > best_acc1:
    selected_checkpoint = BEST_MODEL_PHASE2
    selected_phase = "Phase 2: Fine-tuning"
else:
    selected_checkpoint = BEST_MODEL_PHASE1
    selected_phase = "Phase 1: Transfer learning"

print("Selected checkpoint:", selected_checkpoint)
print("Selected phase:", selected_phase)

# ============================================================
# 15. SAVE FINAL MODEL
# ============================================================

best_model = tf.keras.models.load_model(
    selected_checkpoint,
    compile=False
)

best_model.save(FINAL_MODEL)

print("\nFinal model saved successfully:")
print(os.path.abspath(FINAL_MODEL))

# ============================================================
# 16. SAVE TRAINING HISTORY
# ============================================================

training_history = {
    "phase1": history1.history,
    "phase2": history2.history,
    "best_phase1_val_accuracy": float(best_acc1),
    "best_phase2_val_accuracy": float(best_acc2),
    "selected_phase": selected_phase,
    "selected_checkpoint": selected_checkpoint,
    "classes": class_names,
    "image_size": list(IMG_SIZE),
    "batch_size": BATCH_SIZE
}

with open(HISTORY_FILE, "w", encoding="utf-8") as file:
    json.dump(training_history, file, indent=4)

# ============================================================
# 17. TRAINING SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("TRAINING COMPLETE")
print("=" * 55)

print("Dataset:", os.path.abspath(DATASET_DIR))
print("Classes:", class_names)
print(f"Best Phase 1 validation accuracy: {best_acc1:.4f}")
print(f"Best Phase 2 validation accuracy: {best_acc2:.4f}")
print("Selected phase:", selected_phase)
print("Final model:", os.path.abspath(FINAL_MODEL))
print("Class names:", os.path.abspath(CLASS_NAMES_FILE))
print("Training history:", os.path.abspath(HISTORY_FILE))

print("\nModel is ready for prediction.")
print("=" * 55)
