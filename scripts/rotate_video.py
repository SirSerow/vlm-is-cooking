"""Rotate an MP4 video. Requires ffmpeg on PATH."""
import argparse
import subprocess
from pathlib import Path

ROTATE = {
    "90": "transpose=1",          # clockwise
    "180": "transpose=1,transpose=1",
    "270": "transpose=2",         # counterclockwise
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--degrees", choices=ROTATE, default="90")
    a = p.parse_args()
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y",
            "-i", str(a.input),
            "-vf", ROTATE[a.degrees],
            "-c:a", "copy",
            str(a.output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()