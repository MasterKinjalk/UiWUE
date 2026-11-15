#!/usr/bin/env bash
# Download and extract PhenoBench v1.1.0 (sugar beet UAV images with semantic, plant-instance and
# leaf-instance masks; CC BY-SA 4.0) into $DATA_DIR/phenobench/. Used to fine-tune the RGB
# plant/leaf instance segmentation model (docs/ML_MODELS.md). 7.6 GB zip.
#
# Usage: scripts/download_phenobench.sh      (DATA_DIR defaults to <repo>/data)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA_DIR:-$ROOT/data}"
DEST="$DATA/phenobench"
ZIP="$DEST/PhenoBench-v110.zip"
URL="https://www.phenobench.org/data/PhenoBench-v110.zip"
MD5="5168bba762053725890478432cdbdb1d"

mkdir -p "$DEST"
if [[ -d "$DEST/PhenoBench/train" ]]; then echo "already extracted: $DEST/PhenoBench"; exit 0; fi

# -C - resumes a partial download.
curl -L --fail --retry 5 --retry-delay 10 -C - -o "$ZIP" "$URL"

echo "checking md5 ..."
got="$(md5sum "$ZIP" | cut -d' ' -f1)"
[[ "$got" == "$MD5" ]] || { echo "md5 mismatch: $got (expected $MD5)"; exit 1; }

echo "extracting ..."
if command -v unzip >/dev/null 2>&1; then unzip -q -o "$ZIP" -d "$DEST"
else "${PYTHON:-python3}" -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$ZIP" "$DEST"; fi
ls "$DEST"
echo "done"
