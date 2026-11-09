import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "medical_cnn.pth"


# =====================================================
# MODEL
# =====================================================

class MedicalCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                32 * 56 * 56,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                2
            )
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

    transforms.Resize(
        (224, 224)
    ),

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

    # -------------------------------------------------
    # Last convolution layer
    # -------------------------------------------------

    target_layer = model.features[3]

    # -------------------------------------------------
    # Forward hook
    # -------------------------------------------------

    def forward_hook(
        module,
        input,
        output
    ):

        nonlocal activations

        activations = output

    # -------------------------------------------------
    # Backward hook
    # -------------------------------------------------

    def backward_hook(
        module,
        grad_input,
        grad_output
    ):

        nonlocal gradients

        gradients = grad_output[0]

    # -------------------------------------------------
    # Register hooks
    # -------------------------------------------------

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    try:

        # -------------------------------------------------
        # Forward pass
        # -------------------------------------------------

        output = model(image_tensor)

        # -------------------------------------------------
        # Clear old gradients
        # -------------------------------------------------

        model.zero_grad()

        # -------------------------------------------------
        # Select predicted class
        # -------------------------------------------------

        score = output[0, predicted_class]

        # -------------------------------------------------
        # Backward pass
        # -------------------------------------------------

        score.backward()

    finally:

        # -------------------------------------------------
        # Remove hooks
        # -------------------------------------------------

        forward_handle.remove()
        backward_handle.remove()

    # -------------------------------------------------
    # Check hooks
    # -------------------------------------------------

    if activations is None or gradients is None:

        return original_image

    # -------------------------------------------------
    # Global average pooling of gradients
    # -------------------------------------------------

    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )

    # -------------------------------------------------
    # Weighted activations
    # -------------------------------------------------

    cam = (
        weights * activations
    ).sum(
        dim=1
    )

    # Remove negative values

    cam = torch.relu(cam)

    # -------------------------------------------------
    # Convert to NumPy
    # -------------------------------------------------

    cam = (
        cam
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    # -------------------------------------------------
    # Normalize CAM
    # -------------------------------------------------

    if cam.max() > 0:

        cam = cam / cam.max()

    # -------------------------------------------------
    # Resize CAM
    # -------------------------------------------------

    cam = cv2.resize(
        cam,
        (
            original_image.shape[1],
            original_image.shape[0]
        )
    )

    # -------------------------------------------------
    # Create heatmap
    # -------------------------------------------------

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # -------------------------------------------------
    # Convert RGB image to BGR
    # -------------------------------------------------

    image_bgr = cv2.cvtColor(
        original_image,
        cv2.COLOR_RGB2BGR
    )

    # -------------------------------------------------
    # Overlay heatmap
    # -------------------------------------------------

    overlay = cv2.addWeighted(
        image_bgr,
        0.65,
        heatmap,
        0.35,
        0
    )

    # -------------------------------------------------
    # Convert BGR back to RGB
    # -------------------------------------------------

    overlay = cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )

    return overlay


# =====================================================
# PREDICT IMAGE
# =====================================================

def predict_image(
    image_path,
    gradcam_path
):

    # -------------------------------------------------
    # Open image
    # -------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    # -------------------------------------------------
    # Original image
    # -------------------------------------------------

    original_image = np.array(
        image
    )

    # -------------------------------------------------
    # Preprocess image
    # -------------------------------------------------

    image_tensor = transform(
        image
    )

    # Add batch dimension

    image_tensor = image_tensor.unsqueeze(0)

    # Move to device

    image_tensor = image_tensor.to(device)

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Predicted class
    # -------------------------------------------------

    predicted_class = predicted.item()

    # -------------------------------------------------
    # Result
    # -------------------------------------------------

    result = classes[
        predicted_class
    ]

    # -------------------------------------------------
    # Confidence percentage
    # -------------------------------------------------

    confidence_value = (
        confidence.item() * 100
    )

    # -------------------------------------------------
    # Generate Grad-CAM
    # -------------------------------------------------

    cam_image = generate_gradcam(
        image_tensor,
        original_image,
        predicted_class
    )

    # -------------------------------------------------
    # Save Grad-CAM
    # -------------------------------------------------

    cam_image = Image.fromarray(
        cam_image
    )

    cam_image.save(
        gradcam_path
    )

    # -------------------------------------------------
    # Return result
    # -------------------------------------------------

    return (
        result,
        confidence_value
    )
