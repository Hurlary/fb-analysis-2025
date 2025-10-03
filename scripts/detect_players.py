
from ultralytics import YOLO
from track_utils import run_tracker
import os

video_dir = "../raw_videos"
output_dir = "../detections"
model_path = "../models/yolov8n.pt"

model = YOLO(model_path)

for video_file in os.listdir(video_dir):
    if not video_file.endswith(".mp4"):
        continue
    video_path = os.path.join(video_dir, video_file)
    detections = model.track(source=video_path, persist=True, tracker="bytetrack.yaml", classes=[0])
    run_tracker(detections, output_dir)
