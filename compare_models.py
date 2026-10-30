
import os
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(
    BASE_DIR, "dataset", "HMU-GC-HE-30K", "all_image"
)

MODEL_FILES = [
    "efficientnet_b3.keras",
    "efficientnet_b3_backup.keras",
    "efficientnet_b3_best.keras",
    "efficientnet_b3_phase1_best.keras",
]

CLASS_NAMES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
IMAGE_SIZE = (300, 300)
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Choose one image from each class folder.
test_images = []

for class_name in CLASS_NAMES:
    folder = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(folder):
        print("Missing folder:", folder)
        continue

    files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(EXTENSIONS)
        and os.path.isfile(os.path.join(folder, f))
    )

    if files:
        test_images.append((
            class_name,
            os.path.join(folder, files[0])
        ))
    else:
        print("No images directly inside:", folder)

if not test_images:
    raise RuntimeError("No test images found.")

# Prepare images using the same RGB/resize/0-255 preprocessing.
prepared_images = []

for actual_class, path in test_images:
    with Image.open(path) as im:
        im = im.convert("RGB").resize(IMAGE_SIZE)
        array = np.asarray(im, dtype=np.float32)

    prepared_images.append((actual_class, path, array))

model_dir = os.path.join(BASE_DIR, "model")

for filename in MODEL_FILES:
    path = os.path.join(model_dir, filename)

    if not os.path.isfile(path):
        print(f"\nSkipping missing model: {filename}")
        continue

    print("\n" + "=" * 60)
    print("MODEL:", filename)

    try:
        model = tf.keras.models.load_model(path, compile=False)

        if model.output_shape[-1] != len(CLASS_NAMES):
            print("Unexpected output shape:", model.output_shape)
            continue

        correct = 0
        total = 0

        for actual_class, image_path, array in prepared_images:
            probs = np.asarray(
                model.predict(array[None, ...], verbose=0)[0]
            )

            predicted = CLASS_NAMES[int(np.argmax(probs))]
            confidence = float(np.max(probs)) * 100

            print(
                f"Actual: {actual_class:3} | "
                f"Predicted: {predicted:3} | "
                f"Top probability: {confidence:.2f}%"
            )

            correct += int(predicted == actual_class)
            total += 1

        print(f"Correct on this small diagnostic sample: {correct}/{total}")

    except Exception as error:
        print("Could not test this model:", repr(error))
