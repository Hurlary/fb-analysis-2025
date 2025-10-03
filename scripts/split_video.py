
import cv2
import os

def split_video(video_path, output_dir, chunk_duration=30):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_chunk = int(chunk_duration * fps)

    count = 0
    chunk = 0

    while cap.isOpened():
        out_path = os.path.join(output_dir, f"chunk_{chunk}.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                              (int(cap.get(3)), int(cap.get(4))))
        for _ in range(frames_per_chunk):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            count += 1
        out.release()
        if count >= total_frames:
            break
        chunk += 1
    cap.release()
