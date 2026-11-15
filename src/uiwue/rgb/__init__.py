"""RGB colour traits, the "visible" baseline (owner: Jack).

Run:  python -m uiwue.rgb.run --data-dir <ADA data>
Out:  derived/rgb/traits_runs.parquet, derived/rgb/catalogue.csv
"""

from uiwue.rgb.features import TRAITS, frame_traits

__all__ = ["TRAITS", "frame_traits"]
