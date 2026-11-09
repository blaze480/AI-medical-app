import os

dataset_path = "dataset/chest_xray"

folders = ["train", "test", "val"]
classes = ["NORMAL", "PNEUMONIA"]

print("===== DATASET INFORMATION =====")

for folder in folders:
    print(f"\n{folder.upper()}")

    for class_name in classes:
        path = os.path.join(dataset_path, folder, class_name)

        if os.path.exists(path):
            files = os.listdir(path)

            # Count image files only
            image_files = [
                file for file in files
                if file.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            print(f"{class_name}: {len(image_files)} images")
        else:
            print(f"{class_name}: FOLDER NOT FOUND")