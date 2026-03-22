#!/usr/bin/env python3
"""Package submission zip for NorgesGruppen."""

import zipfile
from pathlib import Path

BASE = Path(__file__).parent
SUBMISSION_DIR = BASE / "submission"
WEIGHTS = BASE / "runs" / "yolov8l_grocery_v2" / "weights" / "best.onnx"
OUTPUT_ZIP = BASE / "submission.zip"

if not WEIGHTS.exists():
    print(f"ERROR: Weights not found at {WEIGHTS}")
    print("Run export_onnx.py first!")
    exit(1)

# Check weight size
size_mb = WEIGHTS.stat().st_size / (1024 * 1024)
print(f"Weight file: {size_mb:.1f} MB (limit: 420 MB)")

if size_mb > 420:
    print("ERROR: Weight file too large! Need to quantize.")
    exit(1)

# Create zip
with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    # run.py at root
    zf.write(SUBMISSION_DIR / "run.py", "run.py")
    # weights at root
    zf.write(WEIGHTS, "best.onnx")

# Verify
with zipfile.ZipFile(OUTPUT_ZIP) as zf:
    print(f"\nZip contents:")
    for info in zf.infolist():
        print(f"  {info.filename} ({info.file_size / 1024 / 1024:.1f} MB)")

total_size = sum(i.file_size for i in zipfile.ZipFile(OUTPUT_ZIP).infolist())
print(f"\nTotal uncompressed: {total_size / 1024 / 1024:.1f} MB (limit: 420 MB)")
print(f"Zip file: {OUTPUT_ZIP} ({OUTPUT_ZIP.stat().st_size / 1024 / 1024:.1f} MB)")
