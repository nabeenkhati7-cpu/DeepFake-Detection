
import cv2
from pathlib import Path
from face_utils import crop_face

# Input and output folders
RAW_DIR = Path("raw_videos")
OUTPUT_DIR = Path(r"D:\prepared_faces_full")

# Extraction settings
FRAMES_PER_VIDEO = 30
MIN_SAVED_FRAMES = 10
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}


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

        # Reuse an existing valid frame when resuming
        if frame_path.exists() and frame_path.stat().st_size > 0:
            saved += 1
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = cap.read()

        if not success:
            continue

        # OpenCV frame is BGR.
        # crop_face() also expects BGR.
        face = crop_face(frame)

        if face is None or face.size == 0:
            continue

        # Save the BGR image directly.
        # Training can convert it to RGB when loading.
        saved_ok = cv2.imwrite(
            str(frame_path),
            face
        )

        if saved_ok:
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


total_videos = 0
successful_videos = 0
failed_videos = []

for label in ["Original", "Deepfakes"]:
    source_folder = RAW_DIR / label

    if not source_folder.is_dir():
        raise FileNotFoundError(
            f"Missing folder: {source_folder}"
        )

    videos = sorted(
        p for p in source_folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        raise RuntimeError(
            f"No videos found in {source_folder}"
        )

    # Detect duplicate filename stems
    stems = [p.stem for p in videos]

    if len(stems) != len(set(stems)):
        raise RuntimeError(
            f"Duplicate video filenames without unique "
            f"extensions in {source_folder}. "
            f"Resolve these before extracting."
        )

    n = len(videos)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    split_videos = {
        "train": videos[:n_train],
        "val": videos[n_train:n_train + n_val],
        "test": videos[n_train + n_val:],
    }

    for split_name, video_list in split_videos.items():
        print(
            f"\n{label} | {split_name}: "
            f"{len(video_list)} videos",
            flush=True
        )

        for i, video in enumerate(video_list, start=1):
            total_videos += 1

            output_folder = (
                OUTPUT_DIR
                / split_name
                / label
                / video.stem
            )

            ok = extract_video(video, output_folder)

            if ok:
                successful_videos += 1
            else:
                failed_videos.append(str(video))

            print(
                f"[{i}/{len(video_list)}] "
                f"{'OK' if ok else 'FAILED'}: {video.name}",
                flush=True
            )


print("\n========== EXTRACTION SUMMARY ==========")
print(f"Videos processed: {total_videos}")
print(f"Successful videos: {successful_videos}")
print(f"Failed videos: {len(failed_videos)}")
print(f"Output folder: {OUTPUT_DIR}")

if failed_videos:
    print("\nFailed videos:")
    for video in failed_videos:
        print(video)
