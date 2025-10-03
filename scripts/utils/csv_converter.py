# utils/csv_converter.py
import csv
from pathlib import Path

def convert_results_to_csv(output_dir: Path):
    """
    Merge all per-chunk detection CSV files into one `detections.csv`
    inside the output_dir.
    """
    merged_csv = output_dir / "detections.csv"
    with open(merged_csv, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["frame", "x1", "y1", "x2", "y2", "track_id", "label", "conf"])

        # only process .csv files (ignore .mp4, etc.)
        for file in sorted(output_dir.glob("*.csv")):
            if file.name == "detections.csv":
                continue  # skip the merged file itself
            with open(file, "r", newline="") as in_f:
                reader = csv.reader(in_f)
                next(reader, None)  # skip header
                for row in reader:
                    writer.writerow(row)
