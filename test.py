import os
import csv
import random
import argparse
from pathlib import Path

import cv2
import yaml
import numpy as np
import torch
from torch.cuda.amp import autocast
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    balanced_accuracy_score,
)
from albumentations import Compose, PadIfNeeded
from transforms.albu import IsotropicResize
from cross_efficient_vit import CrossEfficientViT

# ============================================================
# DETERMINISTIC SEEDING
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
cv2.setNumThreads(1)

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIRS = [
    r"D:\prepared_faces_full",
    r"D:\prepared_faces_ffpp_complete",
]

CONFIG_PATH = os.path.join(BASE_DIR, "configs", "architecture.yaml")
DEFAULT_MODEL = os.path.join(BASE_DIR, "models", "best_model_perfect_acc_v2.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "test_results")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# DIRECTORY SCANNING
# ============================================================
def find_video_folders(class_folder):
    """Find video folders containing test image frames."""
    found = []
    for root, _, files in os.walk(class_folder):
        images = sorted(
            os.path.join(root, name)
            for name in files
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS
            and os.path.isfile(os.path.join(root, name))
            and os.path.getsize(os.path.join(root, name)) > 0
        )
        if images:
            found.append((root, images))
    return sorted(found, key=lambda item: item[0])


# ============================================================
# FRAME LOADING & PREPROCESSING (STRICTLY ALIGNED)
# ============================================================
def load_test_frames(filenames, count, transform):
    """Uniformly sample frames and apply identical preprocessing."""
    if len(filenames) > count:
        indices = np.linspace(0, len(filenames) - 1, count, dtype=int)
        filenames = [filenames[i] for i in indices]

    frames = []
    for filename in filenames:
        img = cv2.imread(filename)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(image=img)["image"]
        frames.append(img.astype(np.float32))
    return frames


# ============================================================
# TOP-K POOLING WITH EMPTY-LIST GUARD
# ============================================================
def aggregate_top_k(frame_probs, top_ratio=0.10):
    """Average the highest-scoring fraction of valid frames safely."""
    if len(frame_probs) == 0:
        raise ValueError("No valid frame probabilities were provided.")

    scores = np.sort(np.asarray(frame_probs, dtype=float))
    k = max(1, int(np.ceil(len(scores) * top_ratio)))
    return float(np.mean(scores[-k:]))


# ============================================================
# STRICT MODEL LOADING
# ============================================================
def load_model(model_path, config, image_size, device):
    """Load model with strict weight alignment."""
    model = CrossEfficientViT(config=config).to(device)
    model.eval()

    dummy = torch.zeros(1, 3, image_size, image_size, device=device)
    with torch.inference_mode():
        model(dummy)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            checkpoint = checkpoint[key]
            break

    if checkpoint and all(k.startswith("module.") for k in checkpoint):
        checkpoint = {k[len("module."):]: v for k, v in checkpoint.items()}

    model.load_state_dict(checkpoint, strict=True)
    return model


# ============================================================
# MAIN TESTING ROUTINE
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate CrossEfficientViT on Test Split")
    parser.add_argument("--model_path", default=DEFAULT_MODEL, type=str, help="Path to checkpoint .pth")
    parser.add_argument("--frames_per_video", default=30, type=int, help="Frames per video (matches training)")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--top_ratio", default=0.10, type=float, help="Top fraction of frames (0.10 = top 3 of 30)")
    parser.add_argument("--max_videos", default=-1, type=int)
    parser.add_argument("--balance_classes", action="store_true", help="Sample equal 1:1 Real vs Fake")
    parser.add_argument(
        "--threshold",
        default=0.50,
        type=float,
        help="Fixed threshold determined on validation set (prevents test data leakage)",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # Path resolution
    model_path = args.model_path if os.path.isabs(args.model_path) else os.path.join(BASE_DIR, args.model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Checkpoint not found at: {model_path}\n"
            f"If you have not trained 'best_model_perfect_acc_v2.pth' yet, "
            f"pass an existing checkpoint using: --model_path models/best_model_All_v2.pth"
        )

    config_path = CONFIG_PATH if os.path.isabs(CONFIG_PATH) else os.path.join(BASE_DIR, CONFIG_PATH)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    image_size = int(config.get("model", {}).get("image-size", 224))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Identical to train.py preprocessing (cv2.BORDER_CONSTANT)
    transform = Compose([
        IsotropicResize(
            max_side=image_size,
            interpolation_down=cv2.INTER_AREA,
            interpolation_up=cv2.INTER_CUBIC,
        ),
        PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            position="center",
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
        ),
    ])

    model = load_model(model_path, config, image_size, device)
    print(f"Loaded checkpoint with strict=True: {model_path}")
    print(f"Testing parameters: {args.frames_per_video} frames/video | Top-{int(args.top_ratio * 100)}% Pooling")
    print(f"Using fixed evaluation threshold: {args.threshold:.4f}\n")

    results = []

    for dataset_dir in DATA_DIRS:
        test_dir = os.path.join(dataset_dir, "test")
        if not os.path.isdir(test_dir):
            print(f"Warning: Directory '{test_dir}' not found. Skipping.")
            continue

        dataset_name = Path(dataset_dir).name
        print(f"Scanning Test Split: {test_dir}")

        for class_name, label in (("Original", 0), ("Deepfakes", 1)):
            class_dir = os.path.join(test_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            videos = find_video_folders(class_dir)
            if args.max_videos > 0:
                videos = videos[:args.max_videos]

            print(f"  {dataset_name} | {class_name}: {len(videos)} video folders")

            for video_folder, filenames in videos:
                frames = load_test_frames(filenames, args.frames_per_video, transform)
                if not frames:
                    continue

                frame_probs = []
                with torch.inference_mode():
                    for start in range(0, len(frames), args.batch_size):
                        batch_np = np.stack(frames[start:start + args.batch_size])
                        batch = torch.from_numpy(batch_np).permute(0, 3, 1, 2).float().to(device)

                        with autocast():
                            logits = model(batch).reshape(-1)
                            probs = torch.sigmoid(logits)

                        frame_probs.extend(probs.cpu().numpy().tolist())

                if len(frame_probs) == 0:
                    continue

                video_score = aggregate_top_k(frame_probs, top_ratio=args.top_ratio)
                results.append({
                    "dataset": dataset_name,
                    "label": label,
                    "probability_fake": video_score,
                    "video_folder": video_folder,
                    "frames_used": len(frame_probs),
                })

    if not results:
        raise RuntimeError("No test videos evaluated. Check dataset paths and folder contents.")

    # Optional 1:1 Class Balancing
    if args.balance_classes:
        reals = [r for r in results if r["label"] == 0]
        fakes = [r for r in results if r["label"] == 1]
        min_len = min(len(reals), len(fakes))
        random.seed(SEED)
        results = random.sample(reals, min_len) + random.sample(fakes, min_len)
        print(f"\n[BALANCED TEST MODE] Sampled {min_len} Real and {min_len} Fake videos.")

    y_true = np.array([r["label"] for r in results], dtype=int)
    y_prob = np.array([r["probability_fake"] for r in results], dtype=float)

    if len(np.unique(y_true)) < 2:
        raise ValueError(f"Testing requires both classes present. Detected labels: {np.unique(y_true)}")

    test_auc = roc_auc_score(y_true, y_prob)

    # Use the fixed threshold passed from validation (avoids test-set snooping)
    eval_thresh = args.threshold
    y_pred = (y_prob >= eval_thresh).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Informational: compute what test Youden's J would be purely for comparison
    fpr, tpr, threshs = roc_curve(y_true, y_prob)
    unbiased_optimal = float(threshs[np.argmax(tpr - fpr)])

    # Export predictions to CSV
    csv_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "label", "probability_fake", "video_folder", "frames_used"]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n============================================================")
    print("                 FINAL UNBIASED TEST RESULTS                ")
    print("============================================================")
    print(f"Total Videos Evaluated  : {len(results)} (Real: {np.sum(y_true == 0)}, Fake: {np.sum(y_true == 1)})")
    print(f"Probability Summary     : Mean={y_prob.mean():.4f} | Min={y_prob.min():.4f} | Max={y_prob.max():.4f}")
    print(f"Test ROC-AUC            : {test_auc:.4f}")
    print(f"Applied Fixed Threshold : {eval_thresh:.4f} (Validation-derived)")
    print("------------------------------------------------------------")
    print(f"Accuracy                : {acc * 100:.2f}%")
    print(f"Balanced Accuracy       : {bal_acc * 100:.2f}%")
    print(f"Precision               : {prec * 100:.2f}%")
    print(f"Recall (Sensitivity)    : {rec * 100:.2f}%")
    print(f"F1-Score                : {f1:.4f}")
    print(f"Confusion Matrix        : TN={tn} | FP={fp} | FN={fn} | TP={tp}")
    print("------------------------------------------------------------")
    print(f"Test-Optimal Cutoff Ref : {unbiased_optimal:.4f} (For diagnostic reference only)")
    print(f"Predictions saved to    : {csv_path}")
    print("============================================================\n")


if __name__ == "__main__":
    main()