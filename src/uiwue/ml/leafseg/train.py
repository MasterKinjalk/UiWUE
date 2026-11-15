"""Fine-tune Mask2Former (Swin-L, COCO instance) on PhenoBench sugar beet plants + leaves.

Single GPU:
    python -m uiwue.ml.leafseg.train --data data/phenobench/PhenoBench --out runs/leafseg-m2f

Multi-GPU (e.g. 4 x A100):
    accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
        -m uiwue.ml.leafseg.train --data data/phenobench/PhenoBench --out runs/leafseg-m2f

Validation: mean Mask2Former loss on the PhenoBench val split plus mask AP via pycocotools
(class-wise, IoU 0.50:0.95) on a fixed subset.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Subset
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from uiwue.ml.leafseg.data import ID2LABEL, PhenoBenchInstances, collate


@torch.no_grad()
def evaluate(model, processor, loader, accelerator) -> dict:
    """Mask AP per class on the main process (val loader is not sharded)."""
    model.eval()
    gts, dts, images = [], [], []
    ann_id, img_id = 1, 0
    for batch in loader:
        out = model(pixel_values=batch["pixel_values"].to(accelerator.device))
        size = batch["pixel_values"].shape[-2:]
        res = processor.post_process_instance_segmentation(
            out,
            threshold=0.3,
            target_sizes=[tuple(size)] * len(batch["mask_labels"]),
            return_binary_maps=True,
        )
        for r, gm, gc in zip(res, batch["mask_labels"], batch["class_labels"], strict=True):
            img_id += 1
            images.append({"id": img_id, "height": int(size[0]), "width": int(size[1])})
            for m, c in zip(gm.numpy().astype(np.uint8), gc.tolist(), strict=True):
                rle = mask_utils.encode(np.asfortranarray(m))
                rle["counts"] = rle["counts"].decode()
                gts.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": c + 1,
                        "segmentation": rle,
                        "area": float(m.sum()),
                        "bbox": list(mask_utils.toBbox(rle)),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
            seg = r["segmentation"]
            if seg is None or len(r["segments_info"]) == 0:
                continue
            seg = seg.cpu().numpy().astype(np.uint8)
            for k, info in enumerate(r["segments_info"]):
                rle = mask_utils.encode(np.asfortranarray(seg[k]))
                rle["counts"] = rle["counts"].decode()
                dts.append(
                    {
                        "image_id": img_id,
                        "category_id": int(info["label_id"]) + 1,
                        "segmentation": rle,
                        "score": float(info["score"]),
                    }
                )
    model.train()
    if not dts:
        return {"ap_plant": 0.0, "ap_leaf": 0.0}
    coco = COCO()
    coco.dataset = {
        "images": images,
        "annotations": gts,
        "categories": [{"id": k + 1, "name": v} for k, v in ID2LABEL.items()],
    }
    coco.createIndex()
    det = coco.loadRes(dts)
    m = {}
    for cid, name in ((1, "plant"), (2, "leaf")):
        ev = COCOeval(coco, det, iouType="segm")
        ev.params.catIds = [cid]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        m[f"ap_{name}"] = float(ev.stats[0])
        m[f"ap50_{name}"] = float(ev.stats[1])
    return m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/phenobench/PhenoBench")
    p.add_argument("--model", default="models/m2f-swinl-inst", help="local dir or HF repo id")
    p.add_argument("--out", default="runs/leafseg-m2f")
    p.add_argument("--size", type=int, default=768)
    p.add_argument("--batch-size", type=int, default=4, help="per GPU")
    p.add_argument("--epochs", type=float, default=50)
    p.add_argument("--max-steps", type=int)
    p.add_argument(
        "--num-queries",
        type=int,
        default=200,
        help="Mask2Former queries = max instances per image (the COCO checkpoint has 200; other values re-initialise them)",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--n-eval", type=int, default=200, help="val images used for AP")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Some Mask2Former heads can go unused for a batch; let DDP tolerate it.
    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="tensorboard", project_dir=args.out, kwargs_handlers=[ddp])
    set_seed(args.seed)
    out = Path(args.out)
    if accelerator.is_main_process:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    accelerator.init_trackers("leafseg")

    train_ds = PhenoBenchInstances(args.data, "train", args.size, augment=True)
    val_ds = PhenoBenchInstances(args.data, "val", args.size, augment=False)
    val_sub = Subset(val_ds, list(range(0, len(val_ds), max(1, len(val_ds) // args.n_eval)))[: args.n_eval])
    accelerator.print(f"train {len(train_ds)} images, val {len(val_ds)} (AP on {len(val_sub)})")
    train_dl = DataLoader(
        train_ds,
        args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate,
        drop_last=True,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_dl = DataLoader(val_sub, args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate)

    processor = Mask2FormerImageProcessor.from_pretrained(args.model)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.model,
        id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()},
        num_queries=args.num_queries,
        ignore_mismatched_sizes=True,
    )
    bb = [p_ for n, p_ in model.named_parameters() if ".encoder." in n and "pixel_level_module" in n]
    bb_ids = {id(p_) for p_ in bb}
    rest = [p_ for p_ in model.parameters() if id(p_) not in bb_ids]
    opt = torch.optim.AdamW(
        [{"params": bb, "lr": args.lr * args.backbone_lr_mult}, {"params": rest, "lr": args.lr}],
        weight_decay=0.05,
    )

    steps_per_epoch = math.ceil(len(train_dl) / accelerator.num_processes)
    total = args.max_steps or int(args.epochs * steps_per_epoch)
    warmup = min(500, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / warmup if s < warmup else (1 - (s - warmup) / max(1, total - warmup)) ** 0.9,
    )
    model, opt, train_dl, sched = accelerator.prepare(model, opt, train_dl, sched)

    best, step, t0 = -1.0, 0, time.time()
    model.train()
    while step < total:
        for batch in train_dl:
            outputs = model(
                pixel_values=batch["pixel_values"],
                mask_labels=[m.to(accelerator.device).float() for m in batch["mask_labels"]],
                class_labels=[c.to(accelerator.device) for c in batch["class_labels"]],
            )
            loss = outputs.loss
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                accelerator.log({"train_loss": loss.item(), "lr": sched.get_last_lr()[-1]}, step=step)
                accelerator.print(
                    f"step {step}/{total}  loss {loss.item():.3f}  {(time.time() - t0) / step:.2f} s/step"
                )
            if step % args.eval_every == 0 or step == total:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    m = evaluate(accelerator.unwrap_model(model), processor, val_dl, accelerator)
                    accelerator.print(f"step {step} {m}")
                    accelerator.log(m, step=step)
                    score = m["ap_plant"] + m["ap_leaf"]
                    if score > best:
                        best = score
                        accelerator.unwrap_model(model).save_pretrained(out / "best")
                        processor.save_pretrained(out / "best")
                        (out / "best" / "metrics.json").write_text(json.dumps({"step": step, **m}, indent=2))
                accelerator.wait_for_everyone()
            if step >= total:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(out / "last")
        processor.save_pretrained(out / "last")
    accelerator.end_training()


if __name__ == "__main__":
    main()
