
import csv
import random
import re
import cv2
from pathlib import Path
from collections import defaultdict
from face_utils import crop_face

# --------------------------------------------------
# PATHS
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(
    r"C:\Users\Administrator\Downloads\DeepFake Detection Model\FaceForensics++_C23"
)
CSV_DIR = DATASET_DIR / "csv"
OUTPUT_DIR = Path(r"D:\prepared_faces_ffpp_complete")

FRAMES_PER_VIDEO = 30
MIN_SAVED_FRAMES = 10
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
SEED = 42

# CSV filename, video folder, label
# 0 = Real, 1 = Fake
SOURCES = [
    ("original.csv", "original", 0),
    ("Deepfakes.csv", "deepfakes", 1),
    ("Face2Face.csv", "Face2Face", 1),
    ("FaceShifter.csv", "FaceShifter", 1),
    ("FaceSwap.csv", "FaceSwap", 1),
    ("NeuralTextures.csv", "NeuralTextures", 1),
]

# Deepfakes.csv videos are stored in the existing project raw_videos.
VIDEO_OVERRIDES = {
    "original": PROJECT_DIR / "raw_videos" / "original",
    "deepfakes": PROJECT_DIR / "raw_videos" / "deepfakes",
}


def resolve_video(relative_path, category):
    """Resolve a CSV path to an existing video file."""
    filename = Path(relative_path.replace("\\", "/")).name

    if category in VIDEO_OVERRIDES:
        candidate = VIDEO_OVERRIDES[category] / filename
    else:
        candidate = DATASET_DIR / category / filename

    if candidate.is_file():
        return candidate

    # Try the CSV's relative path directly.
    candidate = DATASET_DIR / relative_path.replace("\\", "/")
    if candidate.is_file():
        return candidate

    return None


def source_ids(filename, category):
    """
    Return source IDs for source-aware splitting.
    For manipulated files such as 000_003.mp4, both IDs are used.
    Original 000.mp4 maps to source ID 000.
    """
    stem = Path(filename).stem

    if category == "original":
        return [stem]

    # Typical manipulated filename: 000_003
    match = re.fullmatch(r"(\d+)_(\d+)", stem)
    if match:
        return [match.group(1), match.group(2)]

    # Unknown naming format: keep this video in its own group.
    return [f"{category}_{stem}"]


def build_source_groups(records):
    """
    Videos sharing any source ID are grouped together.
    Connected source relationships cannot cross dataset splits.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for record in records:
        ids = record["source_ids"]
        for source_id in ids:
            find(source_id)
        for source_id in ids[1:]:
            union(ids[0], source_id)

    groups = defaultdict(list)
    for record in records:
        root = find(record["source_ids"][0])
        groups[root].append(record)

    return list(groups.values())


def assign_splits(records):
    """
    Assign whole source groups to splits.
    Approximate 70/15/15 by video count, without splitting groups.
    """
    rng = random.Random(SEED)
    groups = build_source_groups(records)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    total = len(records)
    targets = {
        "train": total * 0.70,
        "val": total * 0.15,
        "test": total * 0.15,
    }
    assigned = {"train": [], "val": [], "test": []}

    for group in groups:
        # Choose the split most below its target proportion.
        split = max(
            targets,
            key=lambda name: targets[name] - len(assigned[name])
        )
        assigned[split].extend(group)

    return assigned


def extract_video(video_path, output_folder):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Cannot open: {video_path}", flush=True)
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        print(f"No readable frames: {video_path}", flush=True)
        return False

    count = min(FRAMES_PER_VIDEO, total_frames)
    indices = sorted(set(
        round(i * (total_frames - 1) / max(count - 1, 1))
        for i in range(count)
    ))

    output_folder.mkdir(parents=True, exist_ok=True)
    saved = 0

    for index in indices:
        frame_path = output_folder / f"frame_{index:06d}.jpg"

        # Resume only if the expected frame exists and is readable.
        if frame_path.exists() and frame_path.stat().st_size > 0:
            existing = cv2.imread(str(frame_path))
            if existing is not None:
                saved += 1
                continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = cap.read()
        if not success:
            continue

        # OpenCV frames and crop_face both use BGR.
        face = crop_face(frame)
        if face is None or face.size == 0:
            continue

        if cv2.imwrite(str(frame_path), face):
            saved += 1

    cap.release()

    if saved < MIN_SAVED_FRAMES:
        print(
            f"WARNING: {video_path.name}: "
            f"only {saved}/{len(indices)} frames saved",
            flush=True
        )
        return False

    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    missing = []

    # --------------------------------------------------
    # READ CSVs AND RESOLVE VIDEO PATHS
    # --------------------------------------------------
    for csv_name, category, label in SOURCES:
        csv_path = CSV_DIR / csv_name

        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing CSV: {csv_path}")

        found_count = 0

        with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)

            if "File Path" not in (reader.fieldnames or []):
                raise ValueError(
                    f"'File Path' column missing in {csv_name}: "
                    f"{reader.fieldnames}"
                )

            for row in reader:
                relative_path = (row.get("File Path") or "").strip()
                if not relative_path:
                    continue

                video_path = resolve_video(relative_path, category)
                if video_path is None:
                    missing.append((csv_name, relative_path))
                    continue

                records.append({
                    "path": video_path,
                    "category": category,
                    "label": label,
                    "source_ids": source_ids(
                        video_path.name, category
                    ),
                })
                found_count += 1

        print(
            f"{csv_name}: {found_count} videos found",
            flush=True
        )

    if not records:
        raise RuntimeError("No videos found. Check dataset paths and CSVs.")

    # Detect repeated paths within the collection.
    unique = {}
    for record in records:
        key = str(record["path"].resolve()).lower()
        if key not in unique:
            unique[key] = record

    records = list(unique.values())

    print(f"\nTotal unique videos: {len(records)}")
    print(f"Missing CSV paths: {len(missing)}")

    if missing:
        print("First 20 missing paths:")
        for csv_name, path in missing[:20]:
            print(f"  {csv_name}: {path}")

    # --------------------------------------------------
    # SOURCE-AWARE SPLITS
    # --------------------------------------------------
    splits = assign_splits(records)

    for split_name, split_records in splits.items():
        print(
            f"{split_name}: {len(split_records)} videos "
            f"(real={sum(r['label'] == 0 for r in split_records)}, "
            f"fake={sum(r['label'] == 1 for r in split_records)})"
        )

    # Save a manifest so the split assignments are reproducible.
    manifest_path = OUTPUT_DIR / "manifest.csv"
    with manifest_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "split", "video_path", "category",
            "label", "source_ids"
        ])
        for split_name, split_records in splits.items():
            for record in split_records:
                writer.writerow([
                    split_name,
                    str(record["path"]),
                    record["category"],
                    record["label"],
                    ";".join(record["source_ids"]),
                ])

    # --------------------------------------------------
    # EXTRACT FRAMES
    # --------------------------------------------------
    total = 0
    successful = 0
    failed = []

    for split_name, split_records in splits.items():
        for record in split_records:
            total += 1

            output_folder = (
                OUTPUT_DIR
                / split_name
                / ("Original" if record["label"] == 0 else "Deepfakes")
                / record["category"]
                / record["path"].stem
            )

            ok = extract_video(record["path"], output_folder)

            if ok:
                successful += 1
            else:
                failed.append(str(record["path"]))

            print(
                f"[{total}/{len(records)}] "
                f"{'OK' if ok else 'FAILED'}: "
                f"{record['category']}/{record['path'].name}",
                flush=True
            )

    print("\n========== FACEFORENSICS++ SUMMARY ==========")
    print(f"Videos found: {len(records)}")
    print(f"Videos processed: {total}")
    print(f"Successful videos: {successful}")
    print(f"Failed videos: {len(failed)}")
    print(f"Missing CSV paths: {len(missing)}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")

    if failed:
        print("\nFailed videos (first 30):")
        for video in failed[:30]:
            print(video)


if __name__ == "__main__":
    main()
