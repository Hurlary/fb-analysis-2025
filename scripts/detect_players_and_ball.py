
from ultralytics import YOLO
from track_utils import run_tracker
from utils import looks_like_ball
import os

video_dir = "../raw_videos"
output_dir = "../detections"
model_path = "../models/yolov8n.pt"

model = YOLO(model_path)

for video_file in os.listdir(video_dir):
    if not video_file.endswith(".mp4"):
        continue
    video_path = os.path.join(video_dir, video_file)
    results = model.track(source=video_path, persist=True, tracker="bytetrack.yaml")

    for result in results:
        boxes = result.boxes
        filtered = []
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0:
                filtered.append(box)
            elif conf > 0.5 and looks_like_ball(box):
                filtered.append(box)
        result.boxes = filtered

    run_tracker(results, output_dir)
