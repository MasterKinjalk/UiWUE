"""Main track (Jonathan-Work-Plan section 5): pretrained encoders + a fusion head.

    inputs.py    raw images per sensor and run, fixed normalisation, variants, augmentation
    encoders.py  DOFA (RGB and hyperspectral), ImageBind (thermal and depth), LoRA adapters
    embed.py     stage 1: frozen embeddings for every run      python -m uiwue.fusion.embed
    labels.py    y_deficit, severity stage (S0-S3), turgor deviation
    fusion.py    the fusion model: one token per sensor ([e, e - e_baseline]) + [CLS],
                 2-layer transformer, heads for deficit, stage and turgor
    train.py     5 folds grouped by plant; stage 1 (frozen), stages 2-3 (LoRA);
                 evaluation on every sensor subset; results in derived/detect/fusion_*
                                                               python -m uiwue.fusion.train

Needs the ML extras (requirements-ml.txt) and, for real runs, a GPU. See docs/fusion.md.
"""
