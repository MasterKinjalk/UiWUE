# 3D experiment results, recomputed from the committed table

These CSVs let the 3D pages show real results on a machine that does not have
the local pipeline output (`data/derived/threed/results/`). They were made with
the experiment code, unchanged, on the 3D table that is in git (all 216 plants,
2026-11-15):

```bash
python -m uiwue.threed.experiments --runs derived/threed/traits_runs.parquet --out derived/threed/results
```

Where the local results exist, the pages use those instead.

This run was made on the machine that has the local per-frame and paper-parity
tables (`data/derived/threed/`), so E0a and the E1 noise floor use them. On a
fresh clone those two tables are missing, and E0a shows only our own height
model.

The numbers match `docs/threed/RESULTS.md` closely but not exactly. The
committed table keeps one run per scheduled flyover from 2025-08-24 on, so
re-run folders and earlier dates are left out:

| Experiment | Compared with RESULTS.md |
|---|---|
| E0a paper parity | The same (it reads the local parity table). |
| E0b QC agreement | Fewer runs (1,044). `qc_dark` recall is 0.90 here against 0.79 there. |
| E1 noise floor | Noise within about 5 %. All 12 traits are reliable, as there. |
| E3 stress onset | The same onset days (n_points, leaf_area_est, hull_volume and projected_area day 4; height_p95 day 5). |
| E5 turgor | `leaf_area_est` at lag +1 day, pooled W2 r = −0.59 (n = 334) against −0.58 (n = 402) there. |

E0b used the ADA authors' labels from a local clone of their repository
(`external/ada2025`, not in git). Figures (PNG) are not committed, because the
pages do not use them; the command above regenerates them.
