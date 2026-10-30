
from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "efficientnet_b3.keras"

CLASS_NAMES = [
    "ADI", "DEB", "LYM", "MUC",
    "MUS", "NOR", "STR", "TUM"
]

IMAGE_SIZE = (300, 300)


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


def prepare_image(image: Image.Image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)

    # IMPORTANT:
    # This follows your existing prediction code's /255.0
    # preprocessing. Verify it matches train.py exactly.
    array = array / 255.0

    return np.expand_dims(array, axis=0)


def predict_image(image: Image.Image):
    model = load_model()
    batch = prepare_image(image)

    outputs = np.asarray(
        model.predict(batch, verbose=0)
    )[0]

    if outputs.ndim != 1 or len(outputs) != len(CLASS_NAMES):
        raise ValueError(
            "Model output does not match the 8 configured classes."
        )

    # If the model returns logits instead of probabilities,
    # convert them to probabilities.
    if (
        np.any(outputs < 0)
        or np.any(outputs > 1)
        or not np.isclose(outputs.sum(), 1.0, atol=0.02)
    ):
        shifted = outputs - np.max(outputs)
        exp_scores = np.exp(shifted)
        outputs = exp_scores / exp_scores.sum()

    class_index = int(np.argmax(outputs))

    probabilities = {
        name: float(outputs[i])
        for i, name in enumerate(CLASS_NAMES)
    }

    return {
        "predicted_class": CLASS_NAMES[class_index],
        "confidence": float(outputs[class_index]),
        "probabilities": probabilities,
        "class_index": class_index,
    }
