
import os

DATASET_DIR = (
    r"C:\Gastric_Cancer_AI\dataset"
    r"\HMU-GC-HE-30K\all_image"
)

CLASSES = [
    "ADI", "DEB", "LYM", "MUC",
    "MUS", "NOR", "STR", "TUM"
]

IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff"
)

total = 0

print("\nGASTRIC CANCER DATASET CLASS COUNTS")
print("=" * 50)
print(f"Dataset path: {DATASET_DIR}\n")

for class_name in CLASSES:
    folder = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(folder):
        print(f"{class_name}: FOLDER MISSING")
        continue

    count = sum(
        1
        for filename in os.listdir(folder)
        if filename.lower().endswith(IMAGE_EXTENSIONS)
        and os.path.isfile(os.path.join(folder, filename))
    )

    total += count
    print(f"{class_name}: {count:,} images")

print("=" * 50)
print(f"TOTAL IMAGES: {total:,}")
