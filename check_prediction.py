import os
from predict import predict_image, CLASS_NAMES

DATASET_DIR = r"C:\Gastric_Cancer_AI\dataset\HMU-GC-HE-30K\all_image"

print("=" * 60)
print("TESTING PREDICTIONS ON SAMPLES FROM ALL 8 CLASSES")
print("=" * 60)

for class_name in CLASS_NAMES:
    folder = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(folder):
        print(f"Missing folder: {folder}")
        continue

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    if not images:
        print(f"No images found in {folder}")
        continue

    image_path = os.path.join(folder, images[0])
    predicted, confidence, probs = predict_image(image_path)

    print(f"\n[Actual folder: {class_name} | Image: {images[0]}]")
    print(f"-> Predicted Class: {predicted} (Confidence: {confidence * 100:.2f}%)")
    status = "CORRECT" if predicted == class_name else "MISMATCH"
    print(f"Result: {status}")
