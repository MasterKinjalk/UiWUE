"""Download pretrained weights used for fine-tuning into models/<name>/.

Usage:
    python scripts/download_models.py                 # public models only
    python scripts/download_models.py --gated         # also SAM 3 (needs `hf auth login` + accepted licence)
    python scripts/download_models.py --only promptda-vitl

Models (see docs/ML_MODELS.md for why each one is here):
    promptda-vitl   depth-anything/prompt-depth-anything-vitl-hf    340M  Apache-2.0
    promptda-vits   depth-anything/prompt-depth-anything-vits-hf     25M  Apache-2.0
    m2f-swinl-inst  facebook/mask2former-swin-large-coco-instance   216M  (Mask2Former, COCO instance)
    sam3            facebook/sam3 (gated)                           848M  SAM licence
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent

PUBLIC = {
    "promptda-vitl": "depth-anything/prompt-depth-anything-vitl-hf",
    "promptda-vits": "depth-anything/prompt-depth-anything-vits-hf",
    "m2f-swinl-inst": "facebook/mask2former-swin-large-coco-instance",
}
GATED = {
    "sam3": "facebook/sam3",
}
# Skip duplicate weight formats.
IGNORE = ["*.bin", "*.msgpack", "*.h5", "*.onnx", "*.ot"]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--models-dir", default=str(ROOT / "models"))
    p.add_argument("--gated", action="store_true", help="also download gated models (SAM 3)")
    p.add_argument("--only", nargs="+", help="download only these names")
    args = p.parse_args()

    todo = dict(PUBLIC)
    if args.gated:
        todo.update(GATED)
    if args.only:
        todo = {k: v for k, v in {**PUBLIC, **GATED}.items() if k in args.only}

    for name, repo in todo.items():
        dest = Path(args.models_dir) / name
        # Public repos: token=False so a stale local token cannot cause a 401.
        token = None if name in GATED else False
        print(f"{name:15s} <- {repo}")
        snapshot_download(repo, local_dir=dest, ignore_patterns=IGNORE, token=token)
    print("done")


if __name__ == "__main__":
    main()
