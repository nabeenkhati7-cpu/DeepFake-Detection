import csv
import os
import argparse
import collections
import random
from pathlib import Path

import cv2
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve
from progress.bar import ChargingBar

from cross_efficient_vit import CrossEfficientViT
from deepfakes_dataset import DeepFakesDataset
from utils import check_correct, get_n_params, shuffle_dataset
from albumentations import Compose, PadIfNeeded
from transforms.albu import IsotropicResize

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

DATA_DIRS = [
    r"D:\prepared_faces_full",
    r"D:\prepared_faces_ffpp_complete",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, "models")


# ============================================================
# SMOOTHED BCE LOSS
# ============================================================
class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing=0.05, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        smoothed = targets * (1.0 - 2.0 * self.smoothing) + self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(
            logits, smoothed, pos_weight=self.pos_weight
        )


# ============================================================
# DATASET INDEXING
# ============================================================
def get_training_frames(split_name="train", max_videos=-1, step_frames=2):
    samples = []
    for data_root in DATA_DIRS:
        split_dir = os.path.join(data_root, split_name)
        if not os.path.isdir(split_dir):
            continue

        for class_name, label in (("Original", 0.0), ("Deepfakes", 1.0)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            folder_count = 0
            for root, _, files in os.walk(class_dir):
                frames = sorted(
                    os.path.join(root, f) for f in files
                    if Path(f).suffix.lower() in IMAGE_EXTENSIONS
                    and os.path.isfile(os.path.join(root, f))
                    and os.path.getsize(os.path.join(root, f)) > 0
                )
                if not frames:
                    continue

                if max_videos > 0 and folder_count >= max_videos:
                    break

                if step_frames > 1:
                    frames = frames[::step_frames]

                samples.extend((f, label) for f in frames)
                folder_count += 1
    return samples


def get_validation_videos(max_videos=-1):
    videos = []
    for data_root in DATA_DIRS:
        split_dir = os.path.join(data_root, "val")
        if not os.path.isdir(split_dir):
            continue

        for class_name, label in (("Original", 0), ("Deepfakes", 1)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            v_count = 0
            for root, _, files in os.walk(class_dir):
                frames = sorted(
                    os.path.join(root, f) for f in files
                    if Path(f).suffix.lower() in IMAGE_EXTENSIONS
                    and os.path.isfile(os.path.join(root, f))
                    and os.path.getsize(os.path.join(root, f)) > 0
                )
                if not frames:
                    continue

                if max_videos > 0 and v_count >= max_videos:
                    break

                videos.append((root, frames, label))
                v_count += 1
    return videos


def load_val_frames(filenames, count, transform):
    if len(filenames) > count:
        indices = np.linspace(0, len(filenames) - 1, count, dtype=int)
        filenames = [filenames[i] for i in indices]

    frames = []
    for f in filenames:
        img = cv2.imread(f)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(image=img)["image"]
        frames.append(img.astype(np.float32))
    return frames


# ============================================================
# MAIN TRAINING
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Synchronized CrossEfficientViT Training")
    parser.add_argument("--num_epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--label_smoothing", default=0.05, type=float)
    parser.add_argument("--resume", default="models/best_model_All_v2.pth", type=str)
    parser.add_argument("--dataset", default="perfect_acc_v2", type=str)
    parser.add_argument("--max_videos", default=-1, type=int)
    parser.add_argument("--config", default="configs/architecture.yaml", type=str)
    parser.add_argument("--step_frames", default=2, type=int)
    parser.add_argument("--patience", default=3, type=int)
    # FIX B: Match validation script defaults exactly (30 frames, top 10%)
    parser.add_argument("--val_frames_per_video", default=30, type=int)
    parser.add_argument("--top_ratio", default=0.10, type=float)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # FIX D: Resolve config path against BASE_DIR
    config_path = args.config if os.path.isabs(args.config) else os.path.join(BASE_DIR, args.config)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    image_size = int(config.get("model", {}).get("image-size", 224))
    os.makedirs(MODELS_PATH, exist_ok=True)

    model = CrossEfficientViT(config=config).to(device)

    # Initialize lazy layers
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(2, 3, image_size, image_size, device=device)
        model(dummy)

    # FIX D & Strict Loading: Resolve checkpoint path against BASE_DIR
    if args.resume:
        chk_path = args.resume if os.path.isabs(args.resume) else os.path.join(BASE_DIR, args.resume)
        if not os.path.isfile(chk_path):
            raise FileNotFoundError(f"Checkpoint not found at resolved path: {chk_path}")

        checkpoint = torch.load(chk_path, map_location=device, weights_only=False)
        for key in ("model_state_dict", "state_dict", "model"):
            if isinstance(checkpoint, dict) and key in checkpoint:
                checkpoint = checkpoint[key]
                break

        if checkpoint and all(k.startswith("module.") for k in checkpoint):
            checkpoint = {k[len("module."):]: v for k, v in checkpoint.items()}

        model.load_state_dict(checkpoint, strict=True)
        print(f"Loaded checkpoint with strict=True: {chk_path}")

    # FIX C: Transparent class-weight reporting
    train_samples = get_training_frames("train", max_videos=args.max_videos, step_frames=args.step_frames)
    train_counts = collections.Counter(int(label) for _, label in train_samples)
    real_count = train_counts.get(0, 0)
    fake_count = train_counts.get(1, 0)

    if real_count == 0 or fake_count == 0:
        raise ValueError(f"Training set must contain both classes. Counts: {dict(train_counts)}")

    pos_weight = float(real_count / fake_count)
    print(f"\nTraining Distribution -> Real frames: {real_count} | Fake frames: {fake_count}")
    print(f"Calculated pos_weight (Real/Fake): {pos_weight:.4f}")

    loss_fn = LabelSmoothingBCEWithLogitsLoss(
        smoothing=args.label_smoothing,
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    scaler = GradScaler()

    train_samples = shuffle_dataset(train_samples)
    train_paths = np.asarray([p for p, _ in train_samples])
    train_labels = np.asarray([l for _, l in train_samples], dtype=np.float32)
    train_dataset = DeepFakesDataset(train_paths, train_labels, image_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # FIX A: Matched validation transform using cv2.BORDER_CONSTANT
    val_transform = Compose([
        IsotropicResize(max_side=image_size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=image_size, min_width=image_size, position="center", border_mode=cv2.BORDER_CONSTANT, value=0),
    ])
    val_videos = get_validation_videos(max_videos=args.max_videos)

    best_val_auc = 0.0
    no_improvement = 0
    best_path = os.path.join(MODELS_PATH, f"best_model_{args.dataset}.pth")
    last_path = os.path.join(MODELS_PATH, f"checkpoint_last_{args.dataset}.pth")

    print(f"\nTraining samples: {len(train_samples)} | Validation videos: {len(val_videos)}")
    print(f"Targeting Checkpoint: {best_path}\n")

    for epoch in range(args.num_epochs):
        # ----------------- TRAIN -----------------
        model.train()
        train_loss_sum = 0.0
        train_seen = 0
        train_correct = 0

        bar = ChargingBar(f"Epoch {epoch + 1}/{args.num_epochs}", max=len(train_loader))
        for images, labels in train_loader:
            images = images.permute(0, 3, 1, 2).to(device, non_blocking=True)
            labels = labels.to(device).float().view(-1, 1)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(images)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_count = images.size(0)
            train_loss_sum += loss.item() * batch_count
            train_seen += batch_count

            corrects, _, _ = check_correct(logits.detach(), labels)
            train_correct += corrects
            bar.next()
        bar.finish()

        train_loss = train_loss_sum / max(1, train_seen)
        train_acc = train_correct / max(1, train_seen)

        # ----------------- VIDEO VALIDATION (TOP-10% POOLING) -----------------
        model.eval()
        v_scores = []
        v_labels = []

        with torch.inference_mode():
            for _, frame_paths, label in val_videos:
                frames = load_val_frames(frame_paths, args.val_frames_per_video, val_transform)
                if not frames:
                    continue

                frame_probs = []
                for start in range(0, len(frames), args.batch_size):
                    batch = np.stack(frames[start:start + args.batch_size])
                    batch = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(device)
                    with autocast():
                        logits = model(batch).reshape(-1)
                        probs = torch.sigmoid(logits)
                    frame_probs.extend(probs.cpu().numpy().tolist())

                # FIX B: Top 10% pooling across 30 sampled frames
                scores_arr = np.sort(frame_probs)
                k = max(1, int(np.ceil(len(scores_arr) * args.top_ratio)))
                v_scores.append(float(np.mean(scores_arr[-k:])))
                v_labels.append(label)

        # FIX E: Check for empty validation or single-class evaluation
        if len(v_scores) == 0:
            raise RuntimeError("Validation failed: No video frames could be evaluated.")

        y_true = np.array(v_labels)
        y_prob = np.array(v_scores)

        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            raise ValueError(f"Validation set requires both classes for ROC-AUC. Found classes: {unique_classes}")

        val_auc = roc_auc_score(y_true, y_prob)
        fpr, tpr, threshs = roc_curve(y_true, y_prob)
        best_thresh = threshs[np.argmax(tpr - fpr)]
        bal_acc = balanced_accuracy_score(y_true, (y_prob >= best_thresh).astype(int))

        print(
            f"Epoch {epoch + 1}: Train Loss={train_loss:.4f} | Train Acc={train_acc * 100:.2f}% | "
            f"Video Val AUC={val_auc:.4f} | BalAcc={bal_acc * 100:.2f}% (Cutoff: {best_thresh:.4f})"
        )

        # Single primary metric gate
        if val_auc > best_val_auc:
            print(f"--> Video AUC improved ({best_val_auc:.4f} -> {val_auc:.4f}). Saving checkpoint.")
            best_val_auc = val_auc
            no_improvement = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improvement += 1
            print(f"No AUC gain for {no_improvement} epoch(s). Best AUC: {best_val_auc:.4f}")

        torch.save(model.state_dict(), last_path)
        scheduler.step()

        if no_improvement >= args.patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
            break

    print(f"\nTraining complete. Peak Video ROC-AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()