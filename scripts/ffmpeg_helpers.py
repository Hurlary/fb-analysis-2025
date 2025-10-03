import subprocess
import os
from pathlib import Path

# ✅ Set your FFmpeg path
FFMPEG_PATH = r"C:\Users\DELL\Downloads\ffmpeg\bin\ffmpeg.exe"

def run_ffmpeg_streaming(args):
    """
    Run an ffmpeg command using subprocess and stream logs to console.
    """
    cmd = [FFMPEG_PATH] + args
    print(f"[INFO] Running FFmpeg: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )
        for line in proc.stdout:
            print(line.strip())
        proc.wait()
        return proc.returncode
    except FileNotFoundError:
        raise RuntimeError(f"FFmpeg not found at: {FFMPEG_PATH}")

def split_video(input_path, output_dir, chunk_seconds=30):
    """
    Split a video into smaller chunks.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_pattern = str(output_dir / "chunk_%03d.mp4")

    args = [
        "-i", str(input_path),
        "-c", "copy",
        "-map", "0",
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-reset_timestamps", "1",
        chunk_pattern
    ]

    print(f"[INFO] Splitting raw video -> {output_dir} ({chunk_seconds}s each)")
    code = run_ffmpeg_streaming(args)
    if code != 0:
        raise RuntimeError(f"FFmpeg split failed with code {code}")

def merge_videos_ffmpeg(input_dir, output_path):
    """
    Merge all mp4 files in input_dir into one output file.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    # Create a temporary file list for FFmpeg
    file_list_path = input_dir / "merge_list.txt"
    with open(file_list_path, "w") as f:
        for mp4_file in sorted(input_dir.glob("*.mp4")):
            f.write(f"file '{mp4_file.resolve()}'\n")

    args = [
        "-f", "concat",
        "-safe", "0",
        "-i", str(file_list_path),
        "-c", "copy",
        str(output_path)
    ]

    print(f"[INFO] Merging {len(list(input_dir.glob('*.mp4')))} chunks -> {output_path}")
    code = run_ffmpeg_streaming(args)

    # Clean up temp file
    file_list_path.unlink(missing_ok=True)

    if code != 0:
        raise RuntimeError(f"FFmpeg merge failed with code {code}")
