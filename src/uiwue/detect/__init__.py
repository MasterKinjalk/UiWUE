"""Detection analyses (owner: Jack): onset per signal, sensor benchmark,
ground-truth link, recovery, stress probability per run.

Run:  python -m uiwue.detect.run
Out:  derived/detect/*.parquet
"""

from uiwue.detect.data import load_merged, load_result, trait_columns

__all__ = ["load_merged", "load_result", "trait_columns"]
