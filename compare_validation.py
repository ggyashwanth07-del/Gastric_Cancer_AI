
import os
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(
    BASE_DIR, "dataset", "HMU-GC-HE-30K", "all_image"
)
MODEL_DIR = os.path.join(BASE_DIR, "model")

CLASS_NAMES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
IMG_SIZE = (300, 300)
BATCH_SIZE = 16
SEED = 123

MODEL_FILES = [
    "efficientnet_b3.keras",
    "efficientnet_b3_best.keras",
    "efficientnet_b3_phase1_best.keras",
]

# IMPORTANT: keep shuffle=True for the seeded validation split.
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

if val_ds.class_names != CLASS_NAMES:
    raise ValueError(f"Unexpected class order: {val_ds.class_names}")

print("\nValidation images:", len(val_ds.file_paths))
print("Classes:", val_ds.class_names)

for filename in MODEL_FILES:
    model_path = os.path.join(MODEL_DIR, filename)

    if not os.path.isfile(model_path):
        print("\nSkipping missing model:", filename)
        continue

    print("\n" + "=" * 60)
    print("Evaluating:", filename)

    model = tf.keras.models.load_model(model_path, compile=False)

    all_true = []
    all_pred = []

    # Collect labels and predictions in the SAME iteration.
    # This prevents shuffled images and labels becoming misaligned.
    for images, labels in val_ds:
        probabilities = model(images, training=False).numpy()
        predictions = np.argmax(probabilities, axis=1)

        all_true.extend(labels.numpy().tolist())
        all_pred.extend(predictions.tolist())

    y_true = np.asarray(all_true)
    y_pred = np.asarray(all_pred)

    accuracy = np.mean(y_true == y_pred)
    print(f"Validation accuracy: {accuracy * 100:.2f}%")

    matrix = np.zeros((8, 8), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print("Labels:", CLASS_NAMES)
    print(matrix)

    print("\nPer-class metrics:")
    print(f"{'Class':<8}{'Precision':>12}{'Recall':>10}{'F1':>10}{'Support':>10}")

    for i, name in enumerate(CLASS_NAMES):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        support = matrix[i, :].sum()

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        print(
            f"{name:<8}{precision:>11.3f}"
            f"{recall:>10.3f}{f1:>10.3f}{support:>10}"
        )

    del model
    tf.keras.backend.clear_session()
