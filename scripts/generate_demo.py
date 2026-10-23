"""Regenerate README images with the detector's unchanged default parameters."""

import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plate_detector import process_image, write_image_unicode


def tile(image, title, width=400, height=260):
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    h, w = image.shape[:2]
    scale = min((width - 16) / w, (height - 42) / h)
    resized = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
    rh, rw = resized.shape[:2]
    x, y = (width - rw) // 2, 34 + (height - 42 - rh) // 2
    canvas[y:y + rh, x:x + rw] = resized
    cv2.putText(canvas, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (35, 35, 35), 1)
    return canvas


def save_image(path, image):
    if not write_image_unicode(str(path), image):
        raise RuntimeError(f"Could not encode {path}")


def main():
    output_dir = ROOT / "docs" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = sorted(p for p in (ROOT / "samples").iterdir()
                     if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    if not samples:
        raise RuntimeError("No sample images found")

    rows, records = [], []
    for sample in samples:
        result = process_image(str(sample))
        save_image(output_dir / f"{sample.stem}_detection.jpg", result["best_result"])
        save_image(output_dir / f"{sample.stem}_crop.jpg", result["plate_crop"])
        rows.append(np.concatenate([
            tile(result["src"], f"{sample.stem} | Input"),
            tile(result["best_result"], "Selected candidate"),
            tile(result["plate_crop"], "Extracted region"),
        ], axis=1))
        records.append({
            "sample": sample.name,
            "plate_found": bool(result["plate_found"]),
            "candidate_count": int(result["candidate_count"]),
            "best_score": float(result["best_score"]),
            "best_char_count": int(result["best_char_count"]),
        })

    save_image(output_dir / "overview.jpg", np.concatenate(rows, axis=0))
    (output_dir / "demo_results.json").write_text(
        json.dumps({"parameters": "PlateParams defaults; no per-image tuning",
                    "note": "Candidate selection is not a ground-truth accuracy evaluation.",
                    "results": records}, indent=2) + "\n", encoding="utf-8")
    print(f"Generated demo images for {len(records)} samples in {output_dir}")


if __name__ == "__main__":
    main()
