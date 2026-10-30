import numpy as np
from PIL import Image
from ood_detector import check_ood_image


def validate_histopathology_image(image, filename=None):
    """
    Validate that an uploaded image is an authentic H&E-stained gastric
    histopathology tissue specimen, rejecting AI-generated, synthetic,
    or out-of-distribution non-medical images.
    """
    try:
        accepted, message, ood_score = check_ood_image(image, filename=filename)
        return accepted, message
    except Exception as e:
        return False, f"Validation error: {e}"
