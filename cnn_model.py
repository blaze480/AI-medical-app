import torch
import torch.nn as nn


class MedicalCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # First convolution
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Second convolution
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


# Create the model
model = MedicalCNN()

print(model)


# Test with one fake image
test_image = torch.randn(1, 3, 224, 224)

output = model(test_image)

print("\nInput shape:", test_image.shape)
print("Output shape:", output.shape)
print("Output:", output)