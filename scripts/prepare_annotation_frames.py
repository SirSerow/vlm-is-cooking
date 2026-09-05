"""Sample episode frames with original timestamps using FFmpeg; no ML dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for video in sorted(a.source.glob("[0-9][0-9][0-9]_*.mp4")):
        folder = a.output / "images" / video.stem
        folder.mkdir(parents=True, exist_ok=True)
        with video.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
            "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(video)]))
        timestamps = {round(float(f["best_effort_timestamp_time"]), 4): i
                      for i, f in enumerate(probe["frames"]) if "best_effort_timestamp_time" in f}
        run = subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", str(video),
            "-vf", "select=isnan(prev_selected_t)+gte(t-prev_selected_t\\,1),showinfo",
            "-fps_mode", "vfr", "-q:v", "3", str(folder / "%06d.jpg")],
            capture_output=True, text=True, check=True)
        pts = re.findall(r"Parsed_showinfo[^\n]*\bn:\s*\d+[^\n]*pts_time:([\d.]+)", run.stderr)
        files = sorted(folder.glob("*.jpg"))
        if len(pts) != len(files) or not files:
            raise RuntimeError(f"Frame/timestamp mismatch: {video}")
        for image, timestamp in zip(files, pts):
            t = float(timestamp)
            nearest = min(timestamps, key=lambda x: abs(x-t))
            if abs(nearest-t) > 0.002:
                raise RuntimeError("Cannot map sampled frame to original timestamp")
            manifest.append({"image": image.relative_to(a.output).as_posix(),
                "source_video": video.name, "source_sha256": digest,
                "frame_index": timestamps[nearest], "timestamp_seconds": nearest,
                "session_id": "cooking-session-001", "split": "unassigned",
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest()})
        print(f"{video.name}: {len(files)} frames", flush=True)
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Total: {len(manifest)} frames", flush=True)


if __name__ == "__main__":
    main()
