"""Fine-tune Prompt Depth Anything on ADA ToF + RGB (multi-view self-supervision).

Single GPU:
    python -m uiwue.ml.promptda.train --data data/derived/promptda --out runs/promptda-vitl

Multi-GPU (e.g. 4 x A100):
    accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
        -m uiwue.ml.promptda.train --data data/derived/promptda --out runs/promptda-vitl

Loss: L1 between predicted depth (bilinearly sampled at each supervision point) and the ToF
depth of multi-view points, targets clamped to the prompt range (the model's output is a sigmoid
rescaled to the prompt min/max, so values outside it are unreachable).

Validation (held-out plants) reports, on points from *other* depth frames than the prompt
(independent noise): MAE of the model, and MAE of the baseline "bilinear upsampling of the
prompt". The ratio is the improvement over using the raw ToF.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import PromptDepthAnythingForDepthEstimation

from uiwue.ml.promptda.data import PromptDADataset

DEFAULT_VAL = ["W1_A8", "W2_A8"]


def sample_at(depth: torch.Tensor, sup: torch.Tensor) -> torch.Tensor:
    """depth (B, H, W), sup (B, N, 3) with u, v in pixels -> (B, N) bilinear samples."""
    _, h, w = depth.shape
    gx = sup[..., 0] / (w - 1) * 2 - 1
    gy = sup[..., 1] / (h - 1) * 2 - 1
    grid = torch.stack([gx, gy], -1)[:, :, None]  # (B, N, 1, 2)
    return F.grid_sample(depth[:, None], grid, mode="bilinear", align_corners=True)[:, 0, :, 0]


def depth_loss(pred: torch.Tensor, batch: dict) -> torch.Tensor:
    prompt = batch["prompt_depth"].flatten(1)
    lo, hi = prompt.min(1).values[:, None], prompt.max(1).values[:, None]
    target = batch["sup"][..., 2].clamp(lo, hi)
    return (sample_at(pred, batch["sup"]) - target).abs().mean()


HIST_BINS = 400  # 0.5 mm bins up to 200 mm; last bin collects everything above


def _hist(err_m: torch.Tensor) -> torch.Tensor:
    idx = (err_m * 1000 / 0.5).long().clamp(max=HIST_BINS - 1)
    return torch.bincount(idx, minlength=HIST_BINS).double()


def _summary(hist: torch.Tensor, total_err: float, prefix: str) -> dict:
    n = hist.sum().clamp(min=1)
    cdf = hist.cumsum(0) / n

    def centre(q: float) -> float:  # error (mm) at quantile q, bin centre
        b = torch.searchsorted(cdf, torch.tensor(q, dtype=cdf.dtype, device=cdf.device)).item()
        return 0.5 * (b + 0.5)

    return {
        f"{prefix}mae_mm": 1000 * total_err / n.item(),
        f"{prefix}median_mm": centre(0.5),
        f"{prefix}p90_mm": centre(0.9),
        f"{prefix}within10mm": cdf[19].item(),
    }


@torch.no_grad()
def evaluate(model, loader, accelerator) -> dict:
    """Errors on points from depth frames other than the prompt's (independent ToF noise)."""
    model.eval()
    dev = accelerator.device
    h_m, h_b = (
        torch.zeros(HIST_BINS, device=dev, dtype=torch.float64),
        torch.zeros(HIST_BINS, device=dev, dtype=torch.float64),
    )
    s = torch.zeros(2, device=dev, dtype=torch.float64)
    for batch in loader:
        pred = model(
            pixel_values=batch["pixel_values"], prompt_depth=batch["prompt_depth"]
        ).predicted_depth
        h, w = pred.shape[-2:]
        base = F.interpolate(batch["prompt_depth"], (h, w), mode="bilinear", align_corners=False)[
            :, 0
        ]
        other = batch["sup_other"]
        z = batch["sup"][..., 2]
        e_m = (sample_at(pred.float(), batch["sup"]) - z).abs()[other]
        e_b = (sample_at(base.float(), batch["sup"]) - z).abs()[other]
        h_m += _hist(e_m)
        h_b += _hist(e_b)
        s += torch.stack([e_m.double().sum(), e_b.double().sum()])
    h_m, h_b, s = (accelerator.reduce(t, reduction="sum") for t in (h_m, h_b, s))
    model.train()
    return {**_summary(h_m, s[0].item(), "val_"), **_summary(h_b, s[1].item(), "val_baseline_")}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", default="data/derived/promptda")
    p.add_argument("--model", default="models/promptda-vitl", help="local dir or HF repo id")
    p.add_argument("--out", default="runs/promptda-vitl")
    p.add_argument("--val-plants", nargs="+", default=DEFAULT_VAL)
    p.add_argument("--batch-size", type=int, default=4, help="per GPU")
    p.add_argument("--epochs", type=float, default=5)
    p.add_argument("--max-steps", type=int, help="overrides --epochs")
    p.add_argument("--lr-backbone", type=float, default=5e-6)
    p.add_argument("--lr-head", type=float, default=5e-5)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--n-points", type=int, default=40_000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    accelerator = Accelerator(log_with="tensorboard", project_dir=args.out)
    set_seed(args.seed)
    out = Path(args.out)
    if accelerator.is_main_process:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    accelerator.init_trackers("promptda")

    val_plants = set(args.val_plants)
    train_ds = PromptDADataset(args.data, exclude=val_plants, n_points=args.n_points, augment=True)
    val_ds = PromptDADataset(args.data, plants=val_plants, n_points=args.n_points, augment=False)
    accelerator.print(
        f"train {len(train_ds)} samples, val {len(val_ds)} samples ({sorted(val_plants)})"
    )
    train_dl = DataLoader(
        train_ds,
        args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_dl = DataLoader(
        val_ds, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )

    model = PromptDepthAnythingForDepthEstimation.from_pretrained(args.model)
    backbone = [p_ for n, p_ in model.named_parameters() if n.startswith("backbone.")]
    rest = [p_ for n, p_ in model.named_parameters() if not n.startswith("backbone.")]
    if args.freeze_backbone:
        for p_ in backbone:
            p_.requires_grad_(False)
        groups = [{"params": rest, "lr": args.lr_head}]
    else:
        groups = [
            {"params": backbone, "lr": args.lr_backbone},
            {"params": rest, "lr": args.lr_head},
        ]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)

    steps_per_epoch = math.ceil(len(train_dl) / accelerator.num_processes)
    total = args.max_steps or int(args.epochs * steps_per_epoch)
    warmup = min(500, total // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    model, opt, train_dl, val_dl, sched = accelerator.prepare(model, opt, train_dl, val_dl, sched)

    if len(val_ds):
        m = evaluate(model, val_dl, accelerator)
        accelerator.print(f"step 0 (pretrained) {m}")
        accelerator.log(m, step=0)
    best = float("inf")
    step, t0 = 0, time.time()
    model.train()
    while step < total:
        for batch in train_dl:
            with accelerator.accumulate(model):
                pred = model(
                    pixel_values=batch["pixel_values"], prompt_depth=batch["prompt_depth"]
                ).predicted_depth
                loss = depth_loss(pred.float(), batch)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                accelerator.log(
                    {"train_l1_mm": 1000 * loss.item(), "lr": sched.get_last_lr()[-1]}, step=step
                )
                accelerator.print(
                    f"step {step}/{total}  l1 {1000 * loss.item():.2f} mm  "
                    f"{(time.time() - t0) / step:.2f} s/step"
                )
            if len(val_ds) and (step % args.eval_every == 0 or step == total):
                m = evaluate(model, val_dl, accelerator)
                accelerator.print(f"step {step} {m}")
                accelerator.log(m, step=step)
                if m["val_mae_mm"] < best:
                    best = m["val_mae_mm"]
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        accelerator.unwrap_model(model).save_pretrained(out / "best")
                        (out / "best" / "metrics.json").write_text(
                            json.dumps({"step": step, **m}, indent=2)
                        )
            if step >= total:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(out / "last")
    accelerator.end_training()


if __name__ == "__main__":
    main()
