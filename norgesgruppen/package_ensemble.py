#!/usr/bin/env python3
"""Package ensemble submission zip."""

import zipfile
from pathlib import Path

BASE = Path(__file__).parent
SUBMISSION_DIR = BASE / "submission_ensemble"
OUTPUT_ZIP = BASE / "submission.zip"

# Collect all ONNX files from submission_ensemble dir
models = sorted(SUBMISSION_DIR.glob("*.onnx"))

total_weight = sum(m.stat().st_size for m in models)
print(f"Models: {len(models)}, Total weight: {total_weight / 1024 / 1024:.1f} MB (limit: 420 MB)")

if total_weight > 420 * 1024 * 1024:
    print("ERROR: Total weight exceeds 420 MB!")
    exit(1)
if len(models) > 3:
    print("ERROR: Max 3 weight files!")
    exit(1)

with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(SUBMISSION_DIR / "run.py", "run.py")
    for m in models:
        zf.write(m, m.name)
        print(f"  {m.name}: {m.stat().st_size / 1024 / 1024:.1f} MB")

with zipfile.ZipFile(OUTPUT_ZIP) as zf:
    print(f"\nZip contents:")
    total = 0
    for info in zf.infolist():
        print(f"  {info.filename} ({info.file_size / 1024 / 1024:.1f} MB)")
        total += info.file_size
    print(f"\nTotal uncompressed: {total / 1024 / 1024:.1f} MB")
    print(f"Zip file: {OUTPUT_ZIP} ({OUTPUT_ZIP.stat().st_size / 1024 / 1024:.1f} MB)")
