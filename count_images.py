import os

# Correct location of your images
DATASET_DIR = r"dataset\HMU-GC-HE-30K\all_image"

classes = [
    "ADI",
    "DEB",
    "LYM",
    "MUC",
    "MUS",
    "NOR",
    "STR",
    "TUM"
]

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
)

total = 0

print("\nGASTRIC CANCER DATASET")
print("=" * 45)

for class_name in classes:

    class_path = os.path.join(DATASET_DIR, class_name)

    count = 0

    if os.path.exists(class_path):

        for root, dirs, files in os.walk(class_path):

            for file in files:

                if file.lower().endswith(image_extensions):
                    count += 1

    else:
        print(f"{class_name:10} : FOLDER NOT FOUND")
        continue

    print(f"{class_name:10} : {count}")

    total += count

print("=" * 45)
print(f"TOTAL IMAGES : {total}")
print("=" * 45)