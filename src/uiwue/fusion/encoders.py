"""Pretrained encoders, one per sensor, behind one interface, plus LoRA.

- RGB and hyperspectral: DOFA ViT-B/16 (TorchGeo `dofa_base_patch16_224`, MAE
  weights). Its patch embedding is generated from each band's wavelength, so the
  same model takes 3 webcam bands or all 300 Pika L bands.
- Thermal and depth: ImageBind's dedicated single-channel encoders
  (`imagebind_huge`; its thermal and depth trunks are ViT-S/16, projected to 1024).
  Weights are CC BY-NC 4.0: research use only, stated in docs/fusion.md.

Every encoder maps (B, C, 224, 224) to (B, dim). `encode_frames` averages the
embeddings of a run's frames.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

DOFA_DIM = 768
IMAGEBIND_DIM = 1024


class Encoder(nn.Module):
    sensor: str
    dim: int

    def forward(self, x: torch.Tensor, wavelengths: Sequence[float] | None = None) -> torch.Tensor:
        raise NotImplementedError

    def blocks(self) -> Sequence[nn.Module]:
        """Transformer blocks, in order (LoRA goes on the last few)."""
        raise NotImplementedError

    def encode_frames(self, x: torch.Tensor, wavelengths=None) -> torch.Tensor:
        """x: (B, F, C, H, W) -> (B, dim), mean over frames."""
        b, f = x.shape[:2]
        return self(x.flatten(0, 1), wavelengths).view(b, f, -1).mean(1)


# --------------------------------------------------------------------------- #
# DOFA
# --------------------------------------------------------------------------- #
class DOFAEncoder(Encoder):
    def __init__(self, sensor: str, model: nn.Module | None = None, pretrained: bool = True):
        super().__init__()
        if model is None:
            from torchgeo.models import DOFABase16_Weights, dofa_base_patch16_224

            weights = DOFABase16_Weights.DOFA_MAE if pretrained else None
            model = dofa_base_patch16_224(weights=weights, num_classes=0)
        self.sensor, self.model = sensor, model
        self.dim = model.fc_norm.normalized_shape[0]

    def forward(self, x, wavelengths=None):
        if wavelengths is None:
            raise ValueError("DOFA needs the band wavelengths (µm)")
        return self.model.forward_features(x, list(wavelengths))

    def blocks(self):
        return list(self.model.blocks)


def tiny_dofa(sensor: str = "rgb") -> DOFAEncoder:
    """A small random DOFA for tests (no download)."""
    from torchgeo.models.dofa import DOFA

    return DOFAEncoder(sensor, DOFA(img_size=224, patch_size=16, embed_dim=64, depth=4, num_heads=4, num_classes=0))


# --------------------------------------------------------------------------- #
# ImageBind (thermal, depth)
# --------------------------------------------------------------------------- #
class ImageBindEncoder(Encoder):
    """One ImageBind modality branch: preprocessor -> trunk -> head (no L2 postprocessing,
    so the embedding keeps its magnitude)."""

    def __init__(self, sensor: str, model: nn.Module | None = None):
        super().__init__()
        if sensor not in ("thermal", "depth"):
            raise ValueError(sensor)
        if model is None:
            from imagebind.models import imagebind_model

            model = imagebind_model.imagebind_huge(pretrained=True)
        # Keep only this modality; the rest of imagebind_huge (~1.2 B parameters) is dropped.
        self.sensor = sensor
        self.preprocessor = model.modality_preprocessors[sensor]
        self.trunk = model.modality_trunks[sensor]
        self.head = model.modality_heads[sensor]
        self.dim = IMAGEBIND_DIM
        del model

    def forward(self, x, wavelengths=None):
        v = self.preprocessor(**{self.sensor: x})
        z = self.trunk(**v["trunk"])
        return self.head(z, **v["head"])

    def blocks(self):
        return list(self.trunk.blocks)


def build_encoder(sensor: str, pretrained: bool = True) -> Encoder:
    if sensor in ("rgb", "hsi"):
        return DOFAEncoder(sensor, pretrained=pretrained)
    if sensor in ("thermal", "depth"):
        return ImageBindEncoder(sensor)
    raise ValueError(sensor)


# --------------------------------------------------------------------------- #
# LoRA
# --------------------------------------------------------------------------- #
class LoRALinear(nn.Module):
    """y = W x + (alpha / r) B A x, with W frozen and B initialised to zero (starts as W)."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.a.T @ self.b.T) * self.scale


def add_lora(encoder: Encoder, last_n: int = 4, rank: int = 8, alpha: float = 16.0) -> int:
    """Freeze the encoder and put LoRA on every Linear of its last `last_n` blocks.

    Linears owned by nn.MultiheadAttention are skipped (its forward reads their
    weights directly, so a wrapper would be bypassed). Returns the number wrapped.
    """
    for p in encoder.parameters():
        p.requires_grad_(False)
    n = 0
    for block in encoder.blocks()[-last_n:]:
        skip = {id(m.out_proj) for m in block.modules() if isinstance(m, nn.MultiheadAttention)}
        for parent in list(block.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear) and id(child) not in skip:
                    setattr(parent, name, LoRALinear(child, rank, alpha))
                    n += 1
    return n


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [p for m in module.modules() if isinstance(m, LoRALinear) for p in (m.a, m.b)]
