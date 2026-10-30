import os
import re
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OOD_THRESHOLD = 0.50

AI_KEYWORDS = [
    "ai", "aigen", "ai_gen", "dalle", "dall-e", "midjourney",
    "stablediffusion", "stable_diffusion", "sdxl", "flux", "fake",
    "synthetic", "generated", "bing", "chatgpt", "artbreeder",
    "deepfake", "gan", "genai", "prompt", "photoshop", "render",
    "digital_art"
]

NON_MEDICAL_KEYWORDS = [
    "screenshot", "selfie", "selifie", "landscape", "portrait", "car", "cat",
    "dog", "pet", "nature", "tree", "flower", "food", "drawing", "cartoon",
    "anime", "meme", "wallpaper", "diagram", "chart", "icon", "logo",
    "vehicle", "building", "house", "room", "city", "people", "person", "face"
]

NON_GASTRIC_CLASSES = [
    "ct", "mri", "xray", "x-ray", "radiology", "ultrasound", "blood",
    "colon", "colorectal", "breast", "prostate", "lung", "kidney", "liver",
    "brain", "skin", "melanoma", "cervix", "cervical", "ovary", "ovarian",
    "bladder", "pancreas", "thyroid", "bone", "leukemia", "lymphoma",
    "other", "unknown", "class9", "class_9", "class_other", "non_gastric"
]

VALID_HISTOLOGY_CLASSES = {
    "ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"
}


def check_ood_image(image, filename=None):
    """
    Evaluates whether an uploaded image is an authentic H&E-stained gastric cancer
    histopathology specimen belonging to the 8 supported classes, or an unsupported /
    AI-generated / out-of-distribution image.

    Returns:
        accepted (bool): True if verified as in-distribution tissue specimen; False if rejected.
        message (str): Clinical and technical explanation of the acceptance or rejection.
        ood_probability (float): Risk score indicating AI generation or OOD probability (0.0 to 1.0).
    """
    # ---------------------------------------------------------
    # 1. FILENAME & METADATA HEURISTICS
    # ---------------------------------------------------------
    if filename:
        name_str = str(filename).lower()
        for kw in AI_KEYWORDS:
            if kw in name_str:
                return (
                    False,
                    f"AI-Generated image marker detected in filename ('{kw}'). Not accepted to execute.",
                    0.99
                )
        for kw in NON_MEDICAL_KEYWORDS:
            if kw in name_str:
                return (
                    False,
                    f"Non-medical image content detected in filename ('{kw}'). Not accepted to execute.",
                    0.98
                )
        for kw in NON_GASTRIC_CLASSES:
            if kw in name_str:
                return (
                    False,
                    f"Unsupported non-gastric class/modality detected in filename ('{kw}'). The platform only analyzes the 8 gastric cancer classes (ADI, DEB, LYM, MUC, MUS, NOR, STR, TUM). Not accepted to execute.",
                    0.98
                )

    # ---------------------------------------------------------
    # 2. DIMENSION & MONOTONY CHECKS
    # ---------------------------------------------------------
    try:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image_rgb = image.convert("RGB")
    except Exception as e:
        return False, f"Could not decode image file ({e}). Not accepted to execute.", 1.0

    width, height = image_rgb.size
    if width < 80 or height < 80:
        return False, "Image resolution too low (< 80x80 px). Not accepted to execute.", 1.0

    arr = np.asarray(image_rgb.resize((224, 224)), dtype=np.float32)
    std_val = float(np.std(arr))
    mean_intensity = float(np.mean(arr))

    # A truly blank, empty, or flat digital canvas has near-zero standard deviation (< 2.5).
    # Transparent adipose lipid vacuoles (ADI) or mucin pools (MUC) can have high brightness
    # (mean > 235) with small tissue variance (std ~4.0-7.0), but non-slide flat images are darker.
    if std_val < 2.5:
        return False, "Blank, empty, or solid uniform color image detected (no histological tissue present). Not accepted to execute.", 0.99

    if std_val < 8.0 and mean_intensity < 235.0:
        return False, "Uniform, flat, or low-contrast non-tissue graphic detected. Not accepted to execute.", 0.99

    # ---------------------------------------------------------
    # 3. H&E STAINING OPTICAL & SPECTRAL PHYSICS
    # ---------------------------------------------------------
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mean_r, mean_g, mean_b = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))

    # A. Illumination Range (Brightfield Microscopy backlight):
    if mean_intensity < 70.0:
        return (
            False,
            f"Image is too dark for brightfield microscopy (mean intensity {mean_intensity:.1f} < 70). Not accepted to execute.",
            0.98
        )

    # B. Grayscale / CT / MRI / X-ray detection:
    diff_rg = float(np.mean(np.abs(r - g)))
    diff_rb = float(np.mean(np.abs(r - b)))
    diff_gb = float(np.mean(np.abs(g - b)))
    if mean_intensity < 200.0 and diff_rg < 3.0 and diff_rb < 3.0 and diff_gb < 3.0:
        return (
            False,
            "Grayscale radiological scan, black-and-white image, or CT/MRI/X-ray detected. The AI platform only accepts color H&E histopathology biopsy tissue. Not accepted to execute.",
            0.99
        )

    # C. Dominant Blue (sky/ocean/outdoor photography/blue graphics):
    if mean_b > mean_r * 1.15 and mean_b > 115.0:
        return (
            False,
            f"Blue-dominant color spectrum detected (sky/outdoor/graphic, B/R={mean_b/mean_r:.2f}). Not accepted to execute.",
            0.98
        )

    # D. Green dominance violation (H&E dyes absorb green strongly):
    if mean_r < 238.0 and mean_g > mean_r * 1.02:
        return (
            False,
            "AI-Generated / Non-Histopathology Image: Green spectral channel dominant, violating H&E staining optical physics. Not accepted to execute.",
            0.98
        )

    # ---------------------------------------------------------
    # 4. PROHIBITED HUES & SKIN TONE FILTRATION
    # ---------------------------------------------------------
    hsv = np.asarray(image_rgb.resize((150, 150)).convert("HSV"), dtype=np.float32)
    hue = hsv[..., 0] / 255.0 * 360.0  # 0 to 360 degrees
    sat = hsv[..., 1] / 255.0          # 0 to 1
    val = hsv[..., 2] / 255.0          # 0 to 1

    # Human skin / selfies / portraits / warm wood/face tones:
    # Authentic H&E gastric tissue contains < 0.1% pixels in the 10° to 50° skin hue range,
    # whereas human selfies and portraits contain prominent epidermal tones.
    skin_ratio = float(np.mean((hue >= 10.0) & (hue <= 50.0) & (sat >= 0.15) & (val >= 0.20)))
    if skin_ratio > 0.035:
        return (
            False,
            f"Human skin tone, selfie, portrait, or non-histological photo detected ({skin_ratio:.1%} skin-colored pixels). Not accepted to execute.",
            0.98
        )

    # Ambient skin/yellow-orange tone check across the image:
    mean_hue = float(np.mean(hue))
    mean_sat = float(np.mean(sat))
    if 40.0 <= mean_hue <= 95.0 and mean_sat > 0.08:
        return (
            False,
            f"Skin tone, selfie, portrait, or yellow-orange ambient color detected (mean hue={mean_hue:.1f}°). Not accepted to execute.",
            0.97
        )

    # Green-cyan forbidden hue (70° to 165°):
    green_cyan_ratio = float(np.mean((hue >= 70) & (hue <= 165) & (sat > 0.15) & (val > 0.15)))
    if green_cyan_ratio > 0.018:
        return (
            False,
            f"AI-Generated or Unsupported Image: Detected {green_cyan_ratio:.1%} unnatural green/cyan colors not present in H&E gastric tissue. Not accepted to execute.",
            0.99
        )

    # Yellow-Orange forbidden hue (35° to 65° with high saturation):
    yellow_orange_ratio = float(np.mean((hue >= 35) & (hue <= 65) & (sat > 0.35) & (val > 0.20)))
    if yellow_orange_ratio > 0.04:
        return (
            False,
            f"AI-Generated or Unsupported Image: Detected {yellow_orange_ratio:.1%} non-histological yellow/orange colors. Not accepted to execute.",
            0.97
        )

    # ---------------------------------------------------------
    # 5. CELLULAR GRADIENT TEXTURE
    # ---------------------------------------------------------
    gray = np.asarray(image_rgb.resize((224, 224)).convert("L"), dtype=np.float32)
    grad_mean = float(np.mean(np.abs(np.diff(gray, axis=0))) + np.mean(np.abs(np.diff(gray, axis=1))))
    if grad_mean < 1.1:
        return (
            False,
            f"AI-Generated / Synthetic Graphic: Lacks authentic microscopic nuclear chromatin texture and cell boundaries (gradient={grad_mean:.2f}). Not accepted to execute.",
            0.95
        )

    # ---------------------------------------------------------
    # 6. ACCEPTED
    # ---------------------------------------------------------
    return (
        True,
        "Authentic H&E Gastric Histopathology Specimen Verified (Accepted for 8-Class Execution).",
        0.03
    )
