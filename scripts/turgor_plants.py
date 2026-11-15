"""Print the plant IDs that carry a leaf-thickness (turgor) clip in beds W1 and W2.

Reads the "Sensor Guide" sheet of metadata_rwc_par_en.xlsx (environmental sub-dataset) and prints
space-separated IDs such as "W2_R7 W2_O6 ... W1_B3" for scripts/download_data.sh (Tier 3).
W3 clips are skipped: the work plan uses W1 and W2 only.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOCATION = re.compile(r"^(W[12])\s+([A-R]\d)$")


def turgor_plants(workbook: Path) -> list[str]:
    sheet = pd.read_excel(workbook, sheet_name="Sensor Guide", header=None)
    plants = []
    # Leaf sensor block: rows after the "Leaf Sensors" title until the first empty row.
    in_leaf = False
    for _, row in sheet.iterrows():
        first = str(row[0]).strip()
        if first == "Leaf Sensors":
            in_leaf = True
            continue
        if in_leaf and first in ("nan", ""):
            break
        if in_leaf:
            m = LOCATION.match(str(row[2]).strip())
            if m:
                plants.append(f"{m.group(1)}_{m.group(2)}")
    return plants


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    args = p.parse_args()
    plants = turgor_plants(Path(args.data_dir) / "metadata_rwc_par_en.xlsx")
    if not plants:
        raise SystemExit("no W1/W2 leaf sensors found in the Sensor Guide sheet")
    print(" ".join(plants))


if __name__ == "__main__":
    main()
