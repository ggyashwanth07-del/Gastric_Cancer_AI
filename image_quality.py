
from PIL import Image
import numpy as np


def check_image(image: Image.Image) -> dict:
    """
    Basic technical quality checks.
    These checks flag possible image issues; they do not validate
    whether an image is medically suitable for diagnosis.
    """
    image = image.convert("RGB")
    width, height = image.size
    arr = np.asarray(image, dtype=np.float32)

    issues = []

    if width < 100 or height < 100:
        issues.append("Image resolution is very low.")

    if width != height:
        issues.append("Image is not square; verify expected input format.")

    gray = (
        0.299 * arr[:, :, 0]
        + 0.587 * arr[:, :, 1]
        + 0.114 * arr[:, :, 2]
    )

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Heuristic warning only; thresholds may need calibration
    # against your actual dataset.
    if brightness < 25:
        issues.append("Image may be too dark.")
    elif brightness > 235:
        issues.append("Image may be overexposed.")

    if contrast < 10:
        issues.append("Image has very low contrast.")

    try:
        import cv2

        sharpness = float(
            cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()
        )

        if sharpness < 20:
            issues.append("Image may be blurry.")

    except ImportError:
        sharpness = None

    return {
        "width": width,
        "height": height,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": (
            round(sharpness, 2)
            if sharpness is not None else None
        ),
        "issues": issues,
        "status": "Review recommended" if issues else "Basic checks passed",
    }
