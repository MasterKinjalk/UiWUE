"""Check derived tables against the shared contract (Master-Plan section 6).

    python -m uiwue.core.validate                      # every table under derived/
    python -m uiwue.core.validate derived/x/y.parquet  # one table

Exit code 1 if any table breaks the contract.
"""

import re
import sys
from pathlib import Path

import pandas as pd

from uiwue.core import CATALOGUE_COLUMNS, DERIVED, DESIGN, PREFIXES, SLOTS

MAX_BYTES = 20e6
BOOKKEEPING = ("_capture_hour", "_timestamp_utc", "_n_frames_ok")


def validate(path):
    """List of contract problems in one parquet table (empty list = valid)."""
    path = Path(path)
    df = pd.read_parquet(path)
    problems = []
    if path.stat().st_size > MAX_BYTES:
        problems.append(f"file is {path.stat().st_size / 1e6:.1f} MB; tables in git must stay under 20 MB")
    if not {"bed", "plant"} <= set(df.columns):
        return problems  # a result table (e.g. one row per experiment): no key rules

    keys = [k for k in ("bed", "plant", "date", "hour") if k in df.columns]
    if "hour" in df.columns and "date" not in df.columns:
        problems.append("has `hour` but no `date`")
    if n := df.duplicated(keys).sum():
        problems.append(f"{n} duplicate rows on {keys}")
    if bad := sorted(set(df.bed.astype(str)) - {"W1", "W2", "W3"}):
        problems.append(f"bed values outside W1-W3: {bad[:5]}")
    if bad := sorted({p for p in df.plant.astype(str) if not re.fullmatch(r"[A-R][1-8]", p)}):
        problems.append(f"plant values not like 'J5': {bad[:5]}")
    if "hour" in df.columns and (bad := sorted(set(df.hour) - set(SLOTS))):
        problems.append(f"hour must be the scheduled slot {SLOTS}, found {bad[:5]}")
    if "date" in df.columns and pd.to_datetime(df.date, errors="coerce").isna().any():
        problems.append("date has values that do not parse")

    extra = [c for c in df.columns if c not in keys]
    if bad := [c for c in extra if c in DESIGN]:
        problems.append(f"do not store design columns (added on load): {bad}")
    if bad := [c for c in extra if not c.startswith(PREFIXES)]:
        problems.append(f"columns without a module prefix: {bad[:5]}")

    cat_path = path.parent / "catalogue.csv"
    if not cat_path.exists():
        problems.append(f"missing {cat_path.relative_to(path.parent.parent)}")
        return problems
    cat = pd.read_csv(cat_path)
    if list(cat.columns) != CATALOGUE_COLUMNS:
        problems.append(f"catalogue columns must be {CATALOGUE_COLUMNS}")
        return problems
    if bad := sorted(set(cat.direction) - {-1, 0, 1}):
        problems.append(f"catalogue direction must be -1, 0 or 1, found {bad}")
    traits = [
        c
        for c in extra
        if pd.api.types.is_float_dtype(df[c])
        and not c.endswith("_mad")
        and "qc_" not in c
        and not c.endswith(BOOKKEEPING)
    ]
    if missing := sorted(set(traits) - set(cat.column)):
        problems.append(f"numeric columns missing from catalogue.csv: {missing[:5]}")
    return problems


def main(paths):
    paths = paths or sorted(DERIVED.glob("*/*.parquet"))
    failed = 0
    for p in paths:
        problems = validate(p)
        failed += bool(problems)
        print(("FAIL " if problems else "ok   ") + str(p))
        for msg in problems:
            print("     - " + msg)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
