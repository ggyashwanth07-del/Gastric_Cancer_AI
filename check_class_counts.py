
import os

DATASET_DIR = (
    r"C:\Gastric_Cancer_AI\dataset"
    r"\HMU-GC-HE-30K\all_image"
)

CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

for class_name in CLASSES:
    folder = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(folder):
        print(f"{class_name}: FOLDER MISSING")
        continue

    count = 0
    for root, _, files in os.walk(folder):
        count += sum(
            1 for name in files
            if name.lower().endswith(EXTENSIONS)
        )

    print(f"{class_name}: {count} images")
