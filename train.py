import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader


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
# 4. Fixed train/validation split
# -----------------------------

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)


print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# -----------------------------
# 5. DataLoaders
# -----------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

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


model = MedicalCNN().to(device)


# -----------------------------
# 7. Loss function
# -----------------------------

loss_function = nn.CrossEntropyLoss()


# -----------------------------
# 8. Optimizer
# -----------------------------

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# -----------------------------
# 9. Training
# -----------------------------

epochs = 5

for epoch in range(epochs):

    # Training mode
    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = loss_function(outputs, labels)

        # Clear gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()


    average_loss = total_loss / len(train_loader)


    # -----------------------------
    # Validation
    # -----------------------------

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()


    validation_accuracy = (
        correct / total
    ) * 100


    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"- Loss: {average_loss:.4f} "
        f"- Validation Accuracy: "
        f"{validation_accuracy:.2f}%"
    )


# -----------------------------
# 10. Save model
# -----------------------------

torch.save(
    model.state_dict(),
    "model/medical_cnn.pth"
)

print("\nTraining complete!")
print("Model saved to: model/medical_cnn.pth")