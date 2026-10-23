
import os
import cv2
import torch
import yaml
import numpy as np
from PIL import Image
from torchvision import transforms

from cross_efficient_vit import CrossEfficientViT

# ---------------- CONFIGURATION ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "architecture.yaml")
CHECKPOINT_PATH = os.path.join(
    BASE_DIR, "models", "best_model_All.pth"
)

IMAGE_SIZE = 224
THRESHOLD = 0.5
FRAME_INTERVAL = 5

# -----------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}\n"
        "Wait until training saves best_model_All.pth."
    )

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Read image size from configuration if available
IMAGE_SIZE = int(
    config.get("image-size", config.get("image_size", IMAGE_SIZE))
)

# Build the model
model = CrossEfficientViT(config=config)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]
elif isinstance(checkpoint, dict) and "model" in checkpoint:
    checkpoint = checkpoint["model"]

# Handle normalization buffers that may not exist in this model
model_state = model.state_dict()
for key in ("mean", "std"):
    if key in checkpoint and key not in model_state:
        checkpoint.pop(key)

model.load_state_dict(checkpoint, strict=True)
model.to(device)
model.eval()

# Keep preprocessing consistent with training:
# RGB images, pixel values 0-255; model normalizes internally.
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255.0)
])

@torch.inference_mode()
def predict_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    tensor = preprocess(image).unsqueeze(0).to(device)

    output = model(tensor)
    fake_probability = torch.sigmoid(output.reshape(-1)[0]).item()

    return fake_probability


# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Cannot open webcam. Check camera permissions or try camera index 1."
    )

frame_count = 0
recent_scores = []
display_score = 0.0
display_label = "ANALYZING..."

print("Live video detection started. Press Q to quit.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Unable to read webcam frame.")
            break

        frame_count += 1

        # Predict every few frames to reduce GPU/CPU workload
        if frame_count % FRAME_INTERVAL == 0:
            try:
                score = predict_frame(frame)
                recent_scores.append(score)

                # Smooth predictions over recent sampled frames
                recent_scores = recent_scores[-10:]
                display_score = float(np.mean(recent_scores))

                if display_score >= THRESHOLD:
                    display_label = "POTENTIALLY FAKE"
                else:
                    display_label = "POTENTIALLY REAL"

            except Exception as exc:
                print("Prediction error:", exc)
                display_label = "PREDICTION ERROR"

        # Overlay result
        cv2.putText(
            frame,
            display_label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255) if display_label == "POTENTIALLY FAKE"
            else (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Fake score: {display_score:.1%}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.imshow("Live Video Deepfake Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
