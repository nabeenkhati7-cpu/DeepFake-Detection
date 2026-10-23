import os
import re
import sys
import uuid
import subprocess
from collections import deque

import cv2
import yaml
import torch
import numpy as np
from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
from albumentations import Compose, PadIfNeeded

from transforms.albu import IsotropicResize
from cross_efficient_vit import CrossEfficientViT

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
PHOTO_SCRIPT = os.path.join(BASE_DIR, "predict_photo.py")
VIDEO_SCRIPT = os.path.join(BASE_DIR, "predict_video.py")

CONFIG_PATH = os.path.join(BASE_DIR, "configs", "architecture.yaml")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "cross_efficient_vit.pth")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

PHOTO_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

# --- Load Pretrained Model for Live Streaming ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

image_size = config["model"]["image-size"]

live_model = CrossEfficientViT(config=config)
state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
if "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]
live_model.load_state_dict(state_dict, strict=False)
live_model.to(device)
live_model.eval()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

live_transform = Compose([
    IsotropicResize(max_side=image_size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
    PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT)
])


def allowed_file(filename, extensions):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in extensions
    )


def run_prediction(script_path, file_path, timeout=300):
    """Execute prediction script and extract model predictions."""
    if not os.path.isfile(script_path):
        raise FileNotFoundError(
            f"Prediction script not found: {os.path.basename(script_path)}"
        )

    completed = subprocess.run(
        [sys.executable, script_path],
        input=file_path + "\n",
        capture_output=True,
        text=True,
        cwd=BASE_DIR,
        timeout=timeout,
        errors="replace"
    )

    output = completed.stdout + "\n" + completed.stderr

    if completed.returncode != 0:
        print("Prediction script output:\n", output)
        raise RuntimeError(
            f"{os.path.basename(script_path)} failed. "
            "Check the terminal for details."
        )

    score_match = re.search(
        r"(?:Model output score|Average sigmoid score):\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        output,
        re.IGNORECASE
    )

    prediction_match = re.search(
        r"Prediction:\s*([^\r\n]+)",
        output,
        re.IGNORECASE
    )

    if not score_match or not prediction_match:
        print("Could not parse prediction output:\n", output)
        raise ValueError(
            "Could not read the prediction or score from the script output."
        )

    score = float(score_match.group(1))
    prediction = prediction_match.group(1).strip().upper()

    return score, prediction


def generate_live_frames():
    """Generator for live webcam stream with real-time detection."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    frame_idx = 0
    score_history = deque(maxlen=8)
    current_label = "Detecting..."
    current_color = (255, 255, 0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(80, 80))

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), current_color, 2)

            if frame_idx % 5 == 0:
                face_crop = frame[max(0, y):y + h, max(0, x):x + w]
                if face_crop.size > 0:
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    transformed = live_transform(image=face_rgb)["image"]
                    chw = np.ascontiguousarray(transformed.transpose(2, 0, 1))
                    tensor = torch.from_numpy(chw).float().unsqueeze(0).to(device)

                    with torch.no_grad():
                        score = torch.sigmoid(live_model(tensor)).item()

                    score_history.append(score)
                    avg = sum(score_history) / len(score_history)

                    if avg >= 0.60:
                        current_label = f"FAKE ({avg * 100:.1f}%)"
                        current_color = (0, 0, 255)
                    elif avg >= 0.45:
                        current_label = f"UNCERTAIN ({avg * 100:.1f}%)"
                        current_color = (0, 255, 255)
                    else:
                        current_label = f"REAL ({(1 - avg) * 100:.1f}%)"
                        current_color = (0, 255, 0)

            cv2.putText(frame, current_label, (x, max(25, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        media_type = request.form.get("media_type", "photo").lower()
        upload = request.files.get("file")

        if media_type not in {"photo", "video"}:
            return render_template(
                "index.html", error="Please select photo or video detection."
            )

        if not upload or not upload.filename:
            return render_template(
                "index.html", error="Please select a file to upload."
            )

        extensions = (
            PHOTO_EXTENSIONS if media_type == "photo"
            else VIDEO_EXTENSIONS
        )

        if not allowed_file(upload.filename, extensions):
            allowed = ", ".join(sorted(extensions))
            return render_template(
                "index.html",
                error=f"Unsupported file type. Allowed formats: {allowed}."
            )

        script_path = (
            PHOTO_SCRIPT if media_type == "photo"
            else VIDEO_SCRIPT
        )

        original_name = secure_filename(upload.filename)
        extension = original_name.rsplit(".", 1)[1].lower()
        saved_name = f"{uuid.uuid4().hex}.{extension}"
        file_path = os.path.abspath(
            os.path.join(UPLOAD_FOLDER, saved_name)
        )

        upload.save(file_path)

        try:
            score, prediction = run_prediction(
                script_path,
                file_path,
                timeout=600 if media_type == "video" else 300
            )

            return render_template(
                "result.html",
                prediction=prediction,
                score=score,
                media_type=media_type,
                filename=saved_name
            )

        except subprocess.TimeoutExpired:
            return render_template(
                "index.html",
                error="Prediction timed out. Try a smaller or shorter file."
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print("Prediction error:", exc)
            return render_template(
                "index.html",
                error=str(exc)
            )
        except Exception as exc:
            print("Unexpected prediction error:", exc)
            return render_template(
                "index.html",
                error="An unexpected error occurred. Check the VS Code terminal."
            )

    return render_template("index.html")


@app.route("/live")
def live():
    return render_template("live.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_live_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        error="File is too large. The maximum upload size is 100 MB."
    ), 413


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("🚀 Flask Server running on: http://127.0.0.1:5000")
    print("=" * 55 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)