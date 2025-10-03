import subprocess
from pathlib import Path

def split_video_into_chunks(video_path, output_dir, chunk_length_sec=10):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(output_dir / "chunk_%03d.mp4")
    cmd = [
        r"C:\ffmpeg\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe", "-i", str(video_path),
        "-c", "copy", "-map", "0",
        "-segment_time", str(chunk_length_sec),
        "-f", "segment",
        output_pattern
    ]
    subprocess.run(cmd, check=True)
    return sorted(output_dir.glob("chunk_*.mp4"))