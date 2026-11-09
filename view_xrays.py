import os
from PIL import Image
import matplotlib.pyplot as plt

dataset_path = "dataset/chest_xray"

normal_path = os.path.join(
    dataset_path, "train", "NORMAL"
)

pneumonia_path = os.path.join(
    dataset_path, "train", "PNEUMONIA"
)

# Get the first image from each folder
normal_file = os.listdir(normal_path)[0]
pneumonia_file = os.listdir(pneumonia_path)[0]

normal_image_path = os.path.join(normal_path, normal_file)
pneumonia_image_path = os.path.join(pneumonia_path, pneumonia_file)

# Open images
normal_image = Image.open(normal_image_path)
pneumonia_image = Image.open(pneumonia_image_path)

print("NORMAL image:", normal_file)
print("NORMAL size:", normal_image.size)

print("PNEUMONIA image:", pneumonia_file)
print("PNEUMONIA size:", pneumonia_image.size)

# Display images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(normal_image, cmap="gray")
plt.title("NORMAL")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(pneumonia_image, cmap="gray")
plt.title("PNEUMONIA")
plt.axis("off")

plt.show()