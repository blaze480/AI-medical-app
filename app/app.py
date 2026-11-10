import os
import sys
from pathlib import Path
import uuid

# ==========================================
# PROJECT ROOT
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow importing predictor_backup.py from project root
sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================
# FLASK IMPORTS
# ==========================================

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename


# ==========================================
# AI PREDICTOR
# ==========================================

if os.environ.get("USE_ONNX") == "1":
    from predictor_onnx import predict_image
else:
    from predictor_backup import predict_image


# ==========================================
# FLASK APP
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static")
)

# Maximum upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ==========================================
# FOLDERS
# ==========================================

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)


# ==========================================
# MEDICAL INFORMATION
# ==========================================

MEDICAL_INFO = {

    "NORMAL": {
        "title": "Normal",
        "description": (
            "The AI model did not detect pneumonia-related "
            "abnormalities in the uploaded image."
        ),
        "causes": (
            "No pneumonia-related finding was detected by the model."
        ),
        "guidance": (
            "This AI result is not a medical diagnosis. "
            "Consult a qualified healthcare professional "
            "if you have symptoms or concerns."
        )
    },

    "PNEUMONIA": {
        "title": "Pneumonia",
        "description": (
            "The AI model detected features that may be "
            "associated with pneumonia."
        ),
        "causes": (
            "Pneumonia can be caused by infections including "
            "bacteria, viruses, and fungi."
        ),
        "guidance": (
            "Please consult a qualified healthcare professional "
            "for clinical evaluation and diagnosis."
        )
    }
}


# ==========================================
# ALLOWED IMAGE TYPES
# ==========================================

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    # --------------------------------------
    # CHECK FILE FIELD
    # --------------------------------------

    if "file" not in request.files:
        return "No image file was uploaded.", 400

    file = request.files["file"]

    # --------------------------------------
    # CHECK EMPTY FILE
    # --------------------------------------

    if file.filename == "":
        return "Please select an image.", 400

    # --------------------------------------
    # CHECK FILE EXTENSION
    # --------------------------------------

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        return (
            "Invalid image format. "
            "Please upload JPG, JPEG, PNG, BMP or WEBP.",
            400
        )

    # ======================================
    # UNIQUE FILE NAMES
    # ======================================

    unique_id = uuid.uuid4().hex

    safe_name = secure_filename(file.filename)

    if not safe_name:
        safe_name = "uploaded_image" + extension

    unique_name = unique_id + "_" + safe_name

    result_filename = (
        unique_id +
        "_gradcam" +
        extension
    )

    # ======================================
    # FILE PATHS
    # ======================================

    image_path = UPLOAD_FOLDER / unique_name
    result_path = RESULT_FOLDER / result_filename

    # ======================================
    # SAVE UPLOADED IMAGE
    # ======================================

    try:

        file.save(str(image_path))

    except Exception as e:

        print("File save error:", e)

        return (
            f"Failed to save uploaded image: {e}",
            500
        )

    # ======================================
    # AI PREDICTION
    # ======================================

    try:

        result, confidence = predict_image(
            str(image_path),
            str(result_path)
        )

    except Exception as e:

        print("Prediction error:", e)

        return (
            f"Prediction failed: {e}",
            500
        )

    # ======================================
    # MEDICAL INFORMATION
    # ======================================

    information = MEDICAL_INFO.get(
        result,
        {
            "title": result,
            "description": (
                "No additional information available."
            ),
            "causes": (
                "No additional information available."
            ),
            "guidance": (
                "Please consult a healthcare professional."
            )
        }
    )

    # ======================================
    # IMAGE URLS
    # ======================================

    uploaded_image = (
        "/static/uploads/" +
        unique_name
    )

    if os.environ.get("USE_ONNX") == "1":
        cam_image = None
    else:
        cam_image = (
            "/static/results/" +
            result_filename
        )

    # ======================================
    # CONFIDENCE
    # ======================================

    try:

        confidence_value = round(
            float(confidence),
            1
        )

    except Exception:

        confidence_value = 0.0

    # ======================================
    # RESULT PAGE
    # ======================================

    return render_template(
        "result.html",

        result=result,

        confidence=confidence_value,

        information=information,

        uploaded_image=uploaded_image,

        cam_image=cam_image
    )


# ==========================================
# ERROR HANDLERS
# ==========================================

@app.errorhandler(413)
def file_too_large(error):

    return (
        "File is too large. "
        "Maximum allowed size is 10 MB.",
        413
    )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("AI MEDICAL DIAGNOSIS APPLICATION")
    print("=" * 60)

    print("Project Root :", PROJECT_ROOT)
    print("App Folder   :", BASE_DIR)
    print("Templates    :", BASE_DIR / "templates")
    print("Static       :", BASE_DIR / "static")
    print("Uploads      :", UPLOAD_FOLDER)
    print("Results      :", RESULT_FOLDER)

    print("=" * 60)
    print("Starting Flask server...")
    print("Open: http://127.0.0.1:5000")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
