import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------
# 1. Device
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# -----------------------------
# 2. Image preprocessing
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# -----------------------------
# 3. Load dataset
# -----------------------------

dataset = datasets.ImageFolder(
    "dataset/chest_xray/train",
    transform=transform
)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))


# -----------------------------
# 4. Same train/validation split
# -----------------------------

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)


# -----------------------------
# 5. Validation loader
# -----------------------------

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)


# -----------------------------
# 6. CNN model
# -----------------------------

class MedicalCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(32 * 56 * 56, 128),

            nn.ReLU(),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -----------------------------
# 7. Load trained model
# -----------------------------

model = MedicalCNN().to(device)

model.load_state_dict(
    torch.load(
        "model/medical_cnn.pth",
        map_location=device
    )
)

model.eval()


# -----------------------------
# 8. Make predictions
# -----------------------------

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# -----------------------------
# 9. Accuracy
# -----------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\nValidation Accuracy:")
print(f"{accuracy * 100:.2f}%")


# -----------------------------
# 10. Classification report
# -----------------------------

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=dataset.classes
    )
)


# -----------------------------
# 11. Confusion matrix
# -----------------------------

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)