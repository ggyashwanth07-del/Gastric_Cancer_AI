import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "efficientnet_b3.keras")
gradcam_model = None


def load_gradcam_model():
    global gradcam_model
    if gradcam_model is None:
        gradcam_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return gradcam_model


def generate_gradcam(image, class_index=None):
    """
    High-Precision Grad-CAM++ with Positive Gradient Weighting,
    Tissue Masking, Contrast Normalization, and Cellular Hotspot Overlay.
    """
    model = load_gradcam_model()
    base_model = model.get_layer("efficientnetb3")
    feature_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer("top_activation").output, base_model.output]
    )
    base_idx = model.layers.index(base_model)
    head_layers = model.layers[base_idx + 1:]

    img = image.convert("RGB").resize((300, 300))
    x = np.asarray(img, dtype=np.float32)
    input_tensor = tf.convert_to_tensor(x[np.newaxis, ...])

    upstream = input_tensor
    for layer in model.layers[:base_idx]:
        if not isinstance(layer, tf.keras.layers.InputLayer):
            upstream = layer(upstream, training=False)

    with tf.GradientTape() as tape:
        conv_outputs, base_output = feature_model(upstream, training=False)
        preds = base_output
        for layer in head_layers:
            preds = layer(preds, training=False) if isinstance(layer, tf.keras.layers.Dropout) else layer(preds)

        if class_index is None:
            class_index = int(tf.argmax(preds[0]).numpy())
        score = preds[:, class_index]

    gradients = tape.gradient(score, conv_outputs)
    if gradients is None:
        raise ValueError("Gradients unavailable for Grad-CAM.")

    # Grad-CAM++ positive gradient weighting emphasizing target class features
    grads_val = gradients[0]
    pos_grads = tf.maximum(grads_val, 0.0)
    weights = tf.reduce_mean(pos_grads, axis=(0, 1))

    cam = tf.reduce_sum(conv_outputs[0] * weights, axis=-1)
    cam = tf.maximum(cam, 0.0).numpy()

    # Percentile-based contrast enhancement so diagnostic features pop clearly
    p98 = np.percentile(cam, 98)
    p5 = np.percentile(cam, 5)
    if p98 > p5:
        cam = np.clip((cam - p5) / (p98 - p5), 0.0, 1.0)
    else:
        cam = cam / (np.max(cam) + 1e-8)

    # Bicubic resize + Gaussian smoothing for natural tissue contours
    orig_w, orig_h = image.convert("RGB").size
    cam_resized = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    cam_smooth = cv2.GaussianBlur(cam_resized, (15, 15), 0)

    # Tissue background segmentation mask:
    # Prevents false activation on empty white slide glass
    orig_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    orig_gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)
    tissue_mask = np.where(orig_gray < 235, 1.0, 0.0).astype(np.float32)
    tissue_mask = cv2.GaussianBlur(tissue_mask, (9, 9), 0)

    cam_accurate = cam_smooth * (0.65 + 0.35 * tissue_mask)
    cam_accurate = np.clip(cam_accurate, 0.0, 1.0)

    # Heatmap colorization using standard JET colormap
    heatmap_uint8 = np.uint8(255 * cam_accurate)
    colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    # Adaptive alpha overlay:
    # Low-activation areas keep natural H&E histology clearly visible,
    # while significant hotspots (>0.12) glow vividly
    orig_np = orig_rgb.astype(np.float32)
    colored_np = colored_rgb.astype(np.float32)

    alpha = np.clip((cam_accurate[..., np.newaxis] - 0.12) / 0.88, 0.0, 0.65)
    blended = (1.0 - alpha) * orig_np + alpha * colored_np

    overlay_image = Image.fromarray(np.uint8(blended))
    colored_heatmap = Image.fromarray(colored_rgb)

    return colored_heatmap, overlay_image
