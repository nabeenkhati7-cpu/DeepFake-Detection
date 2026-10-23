
import os
import cv2
import yaml
import torch
import numpy as np

from albumentations import Compose, PadIfNeeded
from transforms.albu import IsotropicResize
from cross_efficient_vit import CrossEfficientViT
from face_utils import crop_face


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(
    BASE_DIR, "configs", "architecture.yaml"
)

# Use the best checkpoint saved by train.py
CHECKPOINT_PATH = os.path.join(
    BASE_DIR, "models", "best_model_All.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MAX_FRAMES = 30
THRESHOLD = 0.50


# ============================================================
# LOAD CONFIGURATION
# ============================================================

if not os.path.isfile(CONFIG_PATH):
    raise FileNotFoundError(
        f"Configuration file not found: {CONFIG_PATH}"
    )

if not os.path.isfile(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Model checkpoint not found: {CHECKPOINT_PATH}\n"
        "Wait for training to save best_model_All.pth."
    )

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

image_size = int(config["model"]["image-size"])

print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] Image size: {image_size}")


# ============================================================
# LOAD MODEL
# ============================================================

model = CrossEfficientViT(config=config)

try:
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )
except TypeError:
    # Compatibility with older PyTorch versions
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]
elif isinstance(checkpoint, dict) and "model" in checkpoint:
    checkpoint = checkpoint["model"]

if not isinstance(checkpoint, dict):
    raise ValueError("The checkpoint is not a valid state dictionary.")

model_state = model.state_dict()

# Remove normalization buffers only if the current model
# does not define them.
for key in ("mean", "std"):
    if key in checkpoint and key not in model_state:
        checkpoint.pop(key)

# Load strictly so architecture mismatches are not hidden.
model.load_state_dict(checkpoint, strict=True)

model.to(DEVICE)
model.eval()

print("[INFO] Model loaded successfully.")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = Compose([
    IsotropicResize(
        max_side=image_size,
        interpolation_down=cv2.INTER_AREA,
        interpolation_up=cv2.INTER_CUBIC
    ),
    PadIfNeeded(
        min_height=image_size,
        min_width=image_size,
        border_mode=cv2.BORDER_CONSTANT
    )
])


# ============================================================
# PREDICT VIDEO
# ============================================================

def predict_video(video_path, max_frames=MAX_FRAMES):

    video_path = os.path.abspath(
        os.path.expanduser(video_path)
    )

    if not os.path.isfile(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] Cannot open the video.")
        return None

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        print("[ERROR] Video contains no readable frames.")
        return None

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        num=min(max_frames, total_frames),
        dtype=int
    )

    scores = []

    print(f"\n[INFO] Video: {video_path}")
    print(f"[INFO] Total frames: {total_frames}")
    print(f"[INFO] Sampling up to {len(frame_indices)} frames\n")

    try:
        for frame_index in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index)
            )

            success, frame = cap.read()

            if not success or frame is None:
                print(
                    f"[WARNING] Could not read frame "
                    f"{frame_index}"
                )
                continue

            # OpenCV BGR -> RGB
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Use the same face preprocessing as training.
            # If training used full frames, remove this crop.
            face = crop_face(frame)

            if face is None or face.size == 0:
                print(
                    f"[WARNING] No usable face in frame "
                    f"{frame_index}; skipping."
                )
                continue

            transformed = transform(image=face)["image"]

            # HWC -> CHW. Preserve 0-255 pixel scale because
            # the model normalizes internally.
            transformed = np.ascontiguousarray(
                transformed.transpose(2, 0, 1)
            )

            tensor = (
                torch.from_numpy(transformed)
                .float()
                .unsqueeze(0)
                .to(DEVICE)
            )

            with torch.inference_mode():
                logits = model(tensor)
                logit = logits.reshape(-1)[0]
                score = torch.sigmoid(logit).item()

            scores.append(score)

            print(
                f"Frame {int(frame_index):6d} | "
                f"Logit: {logit.item(): .6f} | "
                f"Fake score: {score:.4f}"
            )

    finally:
        cap.release()

    if not scores:
        print(
            "\n[ERROR] No frames could be analyzed. "
            "Check face detection and preprocessing."
        )
        return None

    # Aggregate sampled frame scores into one video score.
    average_score = float(np.mean(scores))
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    std_score = float(np.std(scores))

    if average_score >= THRESHOLD:
        prediction = "POTENTIALLY DEEPFAKE"
    else:
        prediction = "POTENTIALLY REAL"

    print("\n" + "=" * 50)
    print("           VIDEO DETECTION RESULT")
    print("=" * 50)

    print(f"Frames analyzed : {len(scores)}")
    print(f"Average fake score: {average_score:.4f}")
    print(f"Minimum frame score: {min_score:.4f}")
    print(f"Maximum frame score: {max_score:.4f}")
    print(f"Score standard deviation: {std_score:.4f}")
    print(f"Threshold: {THRESHOLD:.2f}")
    print(f"Prediction: {prediction}")

    print("=" * 50)
    print(
        "Note: Experimental prediction, not proof "
        "that a video is authentic or manipulated."
    )

    return {
        "prediction": prediction,
        "average_fake_score": average_score,
        "frames_analyzed": len(scores),
        "frame_scores": scores
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    video_path = input(
        "\nEnter the full path of your video: "
    ).strip().strip('"')

    predict_video(video_path)
