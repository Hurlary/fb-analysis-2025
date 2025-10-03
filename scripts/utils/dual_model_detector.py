
from ultralytics import YOLO
from pathlib import Path
import cv2

def detect_on_chunk(args):
    chunk_path, player_model_path, ball_model_path, output_dir = args

    try:
        player_model = YOLO(str(player_model_path))
    except Exception as e:
        print(f"[ERROR] Failed to load player model: {e}")
        return

    try:
        ball_model = YOLO(str(ball_model_path))
    except Exception as e:
        print(f"[ERROR] Failed to load ball model: {e}")
        return

    results_player = player_model(chunk_path)
    results_ball = ball_model(chunk_path)

    output_file = output_dir / (Path(chunk_path).stem + "_results.txt")
    with open(output_file, "w") as f:
        f.write("Player Detection:\n")
        for r in results_player:
            f.write(str(r) + "\n")
        f.write("Ball Detection:\n")
        for r in results_ball:
            f.write(str(r) + "\n")
