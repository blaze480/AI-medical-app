from torchvision import datasets, transforms

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load training dataset
train_dataset = datasets.ImageFolder(
    "dataset/chest_xray/train",
    transform=transform
)

print("Classes:", train_dataset.classes)
print("Number of images:", len(train_dataset))

# Look at the first image
image, label = train_dataset[0]

print("Image shape:", image.shape)
print("Label:", label)
print("Class name:", train_dataset.classes[label])