import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "medical_cnn.onnx"

# Lightweight CPU inference settings
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1

session = ort.InferenceSession(
    str(MODEL_PATH),
    sess_options=options,
    providers=["CPUExecutionProvider"]
)

CLASSES = ["NORMAL", "PNEUMONIA"]


def predict_image(image_path, gradcam_path=None):

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    image_array = np.transpose(
        image_array,
        (2, 0, 1)
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    outputs = session.run(
        None,
        {"input": image_array}
    )

    logits = outputs[0][0]

    exp_values = np.exp(
        logits - np.max(logits)
    )

    probabilities = exp_values / np.sum(exp_values)

    predicted_index = int(np.argmax(probabilities))

    result = CLASSES[predicted_index]

    confidence = float(
        probabilities[predicted_index] * 100
    )

    return result, confidence