# run_dual_yolo_pipeline.py
import argparse
from pathlib import Path
import sys
import cv2
from ultralytics import YOLO
import multiprocessing
import csv

# --- make 'utils' importable
sys.path.append(str(Path(__file__).resolve().parent))

# --- your utils
from utils.draw_ellipse import draw_ellipse
from utils.bytetrack import BYTETracker
from utils.ffmpeg_helpers import split_video, merge_videos_ffmpeg
from utils.csv_converter import convert_results_to_csv

# ========= CONFIG =========
RAW_VIDEO_PATH   = Path(r"C:/Users/DELL/Downloads/For_nate/raw_videos/game1.mp4")
SPLIT_DIR        = Path(r"C:/Users/DELL/Downloads/For_nate/processed_chunks/game1")
OUTPUT_DIR       = Path(r"C:/Users/DELL/Downloads/For_nate/output_results/game1")
FINAL_VIDEO_PATH = OUTPUT_DIR / "full_match_analyzed.mp4"

MODEL_PATH = Path(r"C:/Users/DELL/Downloads/For_nate/scripts/models/best.pt")

SEGMENT_LENGTH_SEC = 30
FORCE_SPLIT = False
UPSCALE_FACTOR = 1
# ==========================

# --- Centralized naming ---
CLASS_MAP = {
    0: "Player",
    1: "Ball",
    2: "Referee"
}


def normalize_classname(name: str) -> str:
    """Map raw YOLO names to class IDs for tracker."""
    n = name.lower()
    if n in ["player", "soccer player", "person", "goalkeeper", "keeper", "goalie"]:
        return 0  # Player
    if n in ["ball", "sports ball", "football", "soccer ball"]:
        return 1  # Ball
    if "ref" in n:
        return 2  # Referee
    return 2  # fallback → referee


def run_detection_worker(video_path: Path, chunk_id: int):
    try:
        print(f"[INFO] Processing chunk {chunk_id} -> {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {video_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path     = OUTPUT_DIR / f"chunk_{chunk_id}_analyzed.mp4"
        video_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

        model   = YOLO(str(MODEL_PATH))
        tracker = BYTETracker(track_thresh=0.2, match_thresh=0.3)

        det_csv = OUTPUT_DIR / f"chunk_{chunk_id}_detections.csv"
        with open(det_csv, "w", newline="") as f:
            csv.writer(f).writerow(["frame", "x1", "y1", "x2", "y2", "track_id", "label", "conf"])

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_up = frame
            if UPSCALE_FACTOR > 1:
                frame_up = cv2.resize(frame, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR,
                                      interpolation=cv2.INTER_LINEAR)

            results = model.predict(frame_up, conf=0.2, imgsz=640, verbose=False)[0]

            detections = []
            if results.boxes is not None and len(results.boxes) > 0:
                for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                    cls = int(cls)
                    raw_name = model.names[cls] if cls in model.names else str(cls)
                    cls_id = normalize_classname(raw_name)

                    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                    if UPSCALE_FACTOR > 1:
                        x1, y1, x2, y2 = x1 / UPSCALE_FACTOR, y1 / UPSCALE_FACTOR, x2 / UPSCALE_FACTOR, y2 / UPSCALE_FACTOR

                    detections.append([x1, y1, x2, y2, float(conf), cls_id])

            # --- run tracker ---
            tracks = tracker.update(detections, frame)

            # --- draw & log ---
            with open(det_csv, "a", newline="") as f:
                csv_writer = csv.writer(f)

                for (x1, y1, x2, y2, tid, cls_id) in tracks:
                    label = f"{CLASS_MAP.get(cls_id, 'Unknown')} {tid}"
                    draw_ellipse(frame, (x1, y1, x2, y2), label)
                    csv_writer.writerow([frame_id, x1, y1, x2, y2, tid, CLASS_MAP.get(cls_id, 'Unknown'), 1.0])

            video_writer.write(frame)
            frame_id += 1

        cap.release()
        video_writer.release()
        print(f"[INFO] Finished chunk {chunk_id}: saved {out_path}")

    except Exception as e:
        print(f"[ERROR] Failed on chunk {chunk_id}: {e}")


def main(args):
    print("[INFO] Starting YOLO football analysis...")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    need_split = FORCE_SPLIT or (len(list(SPLIT_DIR.glob('*.mp4'))) == 0)
    if need_split:
        if not RAW_VIDEO_PATH.exists():
            print(f"[ERROR] Raw video not found: {RAW_VIDEO_PATH}")
            return
        print(f"[INFO] Splitting raw video -> {SPLIT_DIR} (segment={SEGMENT_LENGTH_SEC}s)")
        split_video(RAW_VIDEO_PATH, SPLIT_DIR, SEGMENT_LENGTH_SEC)
    else:
        print("[INFO] Using existing split chunks.")

    video_chunks = sorted(SPLIT_DIR.glob("*.mp4"))
    print(f"[INFO] Found {len(video_chunks)} chunks.")

    if len(video_chunks) == 0:
        print("[WARN] No chunks to process. Exiting.")
        return

    with multiprocessing.Pool(processes=1) as pool:
        pool.starmap(run_detection_worker, [(video, idx) for idx, video in enumerate(video_chunks)])

    print(f"[INFO] Merging analyzed chunks -> {FINAL_VIDEO_PATH}")
    try:
        merge_videos_ffmpeg(OUTPUT_DIR, FINAL_VIDEO_PATH)
    except Exception as e:
        print(f"[WARN] Could not merge chunks automatically: {e}")

    print("[INFO] Exporting YOLO results to CSV...")
    try:
        convert_results_to_csv(OUTPUT_DIR)
        print(f"[INFO] CSV export complete -> {OUTPUT_DIR/'detections.csv'}")
    except Exception as e:
        print(f"[ERROR] CSV conversion failed: {e}")

    print("[INFO] Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
    input("\n[INFO] Press Enter to close this window...")
