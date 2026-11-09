import os
from PIL import Image
import torch
from torchvision import transforms

# Find one NORMAL X-ray
folder = "dataset/chest_xray/train/NORMAL"

filename = os.listdir(folder)[0]
image_path = os.path.join(folder, filename)

# Open image
image = Image.open(image_path).convert("RGB")

print("Original size:", image.size)

# Create preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Preprocess image
image_tensor = preprocess(image)

print("After preprocessing:")
print("Tensor shape:", image_tensor.shape)
print("Minimum pixel value:", image_tensor.min().item())
print("Maximum pixel value:", image_tensor.max().item())