# Fusion run log (2026-11-14/15)

One line per step, so the work can be resumed. Times are local (CST).

- 21:10 Step 1 started: `embed --plants <39> --hsi-windows --workers 8` into `cache/fusion` (log in the session scratchpad). rgb 55 s, thermal 62 s, depth 44 s, 0 failed; hsi running.
- 21:30 Step 2: `train.py` saves `placebo_fold{k}.pt`, the fold's plants, `layers`/`heads`, and LoRA adapters (`lora_state_dict`) per fold; new `predict.py` (external test); tests in `tests/jack/test_fusion_predict.py` pass.
- 21:33 Step 1 done: hsi 3307 runs x 37 variants in 1214 s (8 workers), 38 failed (dark/saturated or unreadable jp2). Total about 23 min. Peak GPU 3.4 GB.
- 21:40 Step 3 done: stage 1 on the GPU server in 38 s (`runs/fusion/fusion/`, pulled; tables validated; committed 419118a). Every setup has bed_effect; thermal's own pre-cutoff score is 0.49.
- 21:41 Step 4 started on the server: `train --stage lora --lora-sensor thermal --workers 6` (log <server work dir>/logs/stage3.log).
- 21:50 Step 5 decision: stage 2 (LoRA on HSI) skipped. Stage-1 HSI early AUC 0.61 [0.42, 0.79], bed_effect, 0.45 on no-clip plants; adding HSI does not change the fused AUC; the benchmark has no HSI features.
- 21:50 External plants: none have thermal/depth/hsi extracted yet (download in progress). Readiness check: scratchpad `ready_plants.py` (zip finished and folder holds every file).
- 21:48 Stage 3 crashed at the very end: `train.write` KeyError 'model' (fusion_auc_by_day had no model column). Fixed (3a0dc82), stage 1 + stage 3 rerun on the server (`runs/logs/run_s1_s3.sh`).
- 22:01 Step 4 done: stage 3 in 454 s; pulled `runs/fusion/fusion+lora_thermal/`, tables validated, committed c32fcb7. Thermal early AUC 0.988 [0.953, 1.000], onset 3.
- 21:45 onwards: external zips are not being extracted in `data/`, so finished zips are extracted (thermal, depth, hsi; dates 24 Aug-22 Sep only) into `cache/raw_external/` by scratchpad `extract_external.py`; RGB is read from the zips in `data/rgb`. A driver (`driver.sh`) extracts and embeds batches into `cache/fusion_external/b<n>` until 07:30; `embedded.txt` lists embedded plants.
- 22:11 External batch b1 (19 W1 plants) embedded in 14 min (hsi 1633 runs, 1 failed). Smoke tests of `predict` on b1 (stage 1: 18 s; LoRA thermal: 244 s) wrote to the scratchpad, not to `derived/`.
- 22:20 Waiting for downloads (thermal is the slowest stream, about 0.6 zips/min). Driver embeds batches of >= 12 newly complete plants.
