# 3D experiment results, recomputed from the committed table

These CSVs let the 3D pages show real results on a machine that does not have
Kinjalk's local pipeline output (`data/derived/threed/results/`). They were
made with his experiment code, unchanged, on the 3D table that is in git:

```bash
python -m uiwue.threed.experiments --runs derived/threed/traits_runs.parquet --out derived/threed/results
```

Where his local results exist, the pages use those instead.

They match `docs/threed/RESULTS.md` closely but not exactly. The committed
table keeps one run per scheduled flyover, so re-run folders are left out, and
the per-frame table and the paper-parity table are not available:

| Experiment | Compared with RESULTS.md |
|---|---|
| E0a paper parity | Only our own height model. The paper-parity rows need the local parity table. |
| E0b QC agreement | Fewer runs (1,044). `qc_dark` recall is 0.90 here against 0.79 there. |
| E1 noise floor | Noise within about 12 %. The same 11 of 12 traits are reliable; `inclination_sd` is not. |
| E3 stress onset | The same onset days (n_points and hull_volume day 6, leaf and projected area day 8). Slopes within 0.3 %/day. |
| E5 turgor | Not run: no leaf-clip plant has a QC-passing run in the committed table. |

E0b used the ADA authors' labels from a local clone of their repository
(`external/ada2025`, not in git). Figures (PNG) are not committed, because the
pages do not use them; the command above regenerates them.
