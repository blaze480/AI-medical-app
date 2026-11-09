import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2


# =====================================================
# MODEL
# =====================================================

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


# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =====================================================
# MODEL PATH
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "medical_cnn.pth"
)


# =====================================================
# LOAD MODEL
# =====================================================

model = MedicalCNN().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model loaded successfully")


# =====================================================
# IMAGE TRANSFORMATION
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# =====================================================
# CLASS NAMES
# =====================================================

classes = [
    "NORMAL",
    "PNEUMONIA"
]


# =====================================================
# GRAD-CAM
# =====================================================

def generate_gradcam(
    image_tensor,
    original_image,
    predicted_class
):

    activations = None
    gradients = None

    target_layer = model.features[3]

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    model.zero_grad()

    output = model(image_tensor)

    score = output[0, predicted_class]

    score.backward()

    forward_handle.remove()
    backward_handle.remove()

    if activations is None or gradients is None:
        return original_image

    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = (
        weights * activations
    ).sum(dim=1)

    cam = torch.relu(cam)

    cam = (
        cam
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    # Normalize
    cam_min = cam.min()
    cam_max = cam.max()

    if cam_max - cam_min != 0:
        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min
        )

    cam = np.uint8(
        cam * 255
    )

    # Resize CAM to original image
    cam = cv2.resize(
        cam,
        (
            original_image.shape[1],
            original_image.shape[0]
        )
    )

    # Heatmap
    heatmap = cv2.applyColorMap(
        cam,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    # Overlay
    overlay = cv2.addWeighted(
        original_image,
        0.6,
        heatmap,
        0.4,
        0
    )

    return overlay


# =====================================================
# PREDICT IMAGE
# =====================================================

def predict_image(
    image_path,
    gradcam_path
):

    image = Image.open(
        image_path
    ).convert("RGB")

    original_image = np.array(
        image
    )

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    # Prediction
    with torch.no_grad():

        output = model(
            image_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = predicted.item()

    result = classes[
        predicted_class
    ]

    confidence_value = (
        confidence.item() * 100
    )

    # Grad-CAM
    cam_image = generate_gradcam(
        image_tensor,
        original_image,
        predicted_class
    )

    cam_image = Image.fromarray(
        cam_image
    )

    # Create output directory if needed
    os.makedirs(
        os.path.dirname(gradcam_path),
        exist_ok=True
    )

    cam_image.save(
        gradcam_path
    )

    return (
        result,
        confidence_value
    )