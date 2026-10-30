
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


ROOT = Path(__file__).resolve().parent
OOD_MODEL_PATH = ROOT / "model" / "ood_detector.keras"

IMAGE_SIZE = (224, 224)

# Starting threshold only. Calibrate it on held-out validation images.
OOD_THRESHOLD = 0.50


@st.cache_resource
def load_ood_model():
    if not OOD_MODEL_PATH.exists():
        raise FileNotFoundError(
            "OOD detector is missing. Run train_ood.py first."
        )

    return tf.keras.models.load_model(OOD_MODEL_PATH)


def check_ood_image(image: Image.Image):
    """
    Returns:
        accepted: True if the detector considers the image in-dataset
        message: explanation
        ood_probability: estimated out-of-dataset probability
    """

    model = load_ood_model()

    image = image.convert("RGB").resize(IMAGE_SIZE)
    pixels = np.asarray(image, dtype=np.float32)

    pixels = tf.keras.applications.mobilenet_v2.preprocess_input(
        pixels
    )

    batch = np.expand_dims(pixels, axis=0)

    ood_probability = float(
        model.predict(batch, verbose=0)[0][0]
    )

    if ood_probability >= OOD_THRESHOLD:
        return (
            False,
            "Out of Dataset / Unsupported Image",
            ood_probability
        )

    return (
        True,
        "Detector accepted the image for further analysis.",
        ood_probability
    )
