"""Pretrained encoders, one per sensor, behind one interface, plus LoRA.

- RGB and hyperspectral: DOFA ViT-B/16 (TorchGeo `dofa_base_patch16_224`, MAE
  weights). Its patch embedding is generated from each band's wavelength, so the
  same model takes 3 webcam bands or all 300 Pika L bands.
- Thermal and depth: ImageBind's dedicated single-channel encoders
  (`imagebind_huge`; thermal trunk ViT-B/16, depth trunk ViT-S/16, both projected to 1024).
  Weights are CC BY-NC 4.0: research use only, stated in docs/fusion.md.

Every encoder maps (B, C, 224, 224) to (B, dim). `encode_frames` averages the
embeddings of a run's frames.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from uiwue import core

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
        """TorchGeo's DOFA.forward_features, with the patch embedding kept in float32.

        DOFA generates its patch-embedding kernels from a sine/cosine code of each
        wavelength in nm (positions up to ~1025). Under bf16 autocast that code is
        computed in bf16 and the kernels come out wrong (30 % median error per band on
        the 300 Pika L bands, verified 2026-11-14), so this step runs with autocast off.
        The transformer blocks still run under the caller's autocast.
        """
        if wavelengths is None:
            raise ValueError("DOFA needs the band wavelengths (µm)")
        m = self.model
        with torch.autocast(device_type=x.device.type, enabled=False):
            waves = torch.tensor(list(wavelengths), device=x.device, dtype=torch.float32)
            x, _ = m.patch_embed(x.float(), waves)
        x = x + m.pos_embed[:, 1:, :]
        cls = (m.cls_token + m.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        for block in m.blocks:
            x = block(x)
        if m.global_pool:  # mean over patch tokens, then fc_norm (not in the MAE checkpoint: identity init)
            return m.fc_norm(x[:, 1:, :].mean(dim=1))
        return m.norm(x)[:, 0]

    def blocks(self):
        return list(self.model.blocks)


def tiny_dofa(sensor: str = "rgb") -> DOFAEncoder:
    """A small random DOFA for tests (no download)."""
    from torchgeo.models.dofa import DOFA

    return DOFAEncoder(sensor, DOFA(img_size=224, patch_size=16, embed_dim=64, depth=4, num_heads=4, num_classes=0))


# --------------------------------------------------------------------------- #
# ImageBind (thermal, depth)
# --------------------------------------------------------------------------- #
IMAGEBIND_URL = "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
# imagebind_huge(pretrained=True) downloads to ".checkpoints/" relative to the working directory,
# so a run started elsewhere downloads 4.8 GB again. The path is fixed here instead.
IMAGEBIND_CKPT = Path(os.environ.get("UIWUE_IMAGEBIND_CKPT", core.REPO / ".checkpoints" / "imagebind_huge.pth"))


def _imagebind_model_module():
    """imagebind.models.imagebind_model, imported without running imagebind/__init__.py.

    The package __init__ imports imagebind.data, which needs pkg_resources (gone from
    setuptools >= 81; the lock pins 84.2), torchaudio and pytorchvideo. The thermal and
    depth branches use none of them, so the bare package is registered and only the
    model modules are imported (they need einops, ftfy, regex, iopath, timm).
    """
    import importlib
    import importlib.util
    import sys
    import types

    if "imagebind" not in sys.modules:
        spec = importlib.util.find_spec("imagebind")
        if spec is None:
            raise ImportError("pip install --no-deps git+https://github.com/facebookresearch/ImageBind")
        pkg = types.ModuleType("imagebind")
        pkg.__path__ = list(spec.submodule_search_locations)
        sys.modules["imagebind"] = pkg
    return importlib.import_module("imagebind.models.imagebind_model")


def load_imagebind(path: Path | str = IMAGEBIND_CKPT) -> nn.Module:
    """imagebind_huge with the real checkpoint, loaded strictly (no missing or unexpected keys)."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        torch.hub.download_url_to_file(IMAGEBIND_URL, str(part), progress=True)
        part.replace(path)
    model = _imagebind_model_module().imagebind_huge(pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True, mmap=True), strict=True)
    return model


class ImageBindEncoder(Encoder):
    """One ImageBind modality branch: preprocessor -> trunk -> head (no L2 postprocessing,
    so the embedding keeps its magnitude). Thermal is a ViT-B/16 trunk (768 wide), depth a
    ViT-S/16 (384 wide); both heads project the [CLS] token to 1024."""

    def __init__(self, sensor: str, model: nn.Module | None = None):
        super().__init__()
        if sensor not in ("thermal", "depth"):
            raise ValueError(sensor)
        if model is None:
            model = load_imagebind()
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
        # On the base layer's device and dtype: LoRAStage adds LoRA after moving the encoder to
        # the GPU, and adapters created on the CPU failed there ("mat2 is on cpu").
        kw = dict(device=base.weight.device, dtype=base.weight.dtype)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, **kw))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, **kw))
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
