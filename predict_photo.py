
import os
import sys
import yaml
import torch
import cv2
import numpy as np

from albumentations import Compose, PadIfNeeded
from transforms.albu import IsotropicResize
from cross_efficient_vit import CrossEfficientViT

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "architecture.yaml")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "best_model_All.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load model configuration
with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

image_size = int(config["model"]["image-size"])

# Create model
model = CrossEfficientViT(config=config)

# Load trained checkpoint
if not os.path.isfile(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

# Support common checkpoint formats
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]
elif isinstance(checkpoint, dict) and "model" in checkpoint:
    checkpoint = checkpoint["model"]

model_state = model.state_dict()

# Handle mean/std buffers for compatible model versions
for key in ("mean", "std"):
    if key in checkpoint and key not in model_state:
        checkpoint.pop(key)
    elif key in model_state and key not in checkpoint:
        checkpoint[key] = model_state[key]

# Strict loading prevents silently ignoring incompatible weights
model.load_state_dict(checkpoint, strict=True)
model.to(device)
model.eval()

print("Loaded checkpoint:", os.path.basename(CHECKPOINT_PATH))

# Match validation resizing and padding
transform = Compose([
    IsotropicResize(
        max_side=image_size,
        interpolation_down=cv2.INTER_AREA,
        interpolation_up=cv2.INTER_CUBIC
    ),
    PadIfNeeded(
        min_height=image_size,
        min_width=image_size,
        position="center",
        border_mode=cv2.BORDER_CONSTANT
    )
])

# Ask for image path
photo_path = input(
    "Enter the full path of your photo: "
).strip().strip('"')

if not os.path.isfile(photo_path):
    print("ERROR: Photo file not found.")
    sys.exit(1)

# Read image
image_bgr = cv2.imread(photo_path)

if image_bgr is None:
    print("ERROR: Cannot read this image.")
    sys.exit(1)

# IMPORTANT:
# Training uses full frames, not cropped faces.
# Convert BGR to RGB, then resize and pad.
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

transformed = transform(image=image_rgb)["image"]

# Convert HWC to CHW, preserving pixel values in the 0-255 range.
image_chw = np.ascontiguousarray(
    transformed.transpose(2, 0, 1)
)

image_tensor = (
    torch.from_numpy(image_chw)
    .float()
    .unsqueeze(0)
    .to(device)
)

# Predict
with torch.no_grad():
    logits = model(image_tensor)
    score = torch.sigmoid(logits).item()

print(f"Raw model output: {logits.item():.4f}")
print(f"Model fake score: {score:.4f}")

if score >= 0.5:
    print("Prediction: POTENTIALLY DEEPFAKE")
else:
    print("Prediction: POTENTIALLY REAL")

print(
    "Warning: Experimental result only; "
    "this is not proof of authenticity."
)