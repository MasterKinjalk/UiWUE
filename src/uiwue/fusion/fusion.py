"""The fusion model (Jonathan-Work-Plan section 5).

    HSI cube    -> DOFA        -> e ┐
    RGB frame   -> DOFA        -> e ├─ each: [z(e(t)), z(e(t) - e(baseline))] -> linear -> 256
    thermal     -> ImageBind-T -> e │        + learned sensor token
    depth       -> ImageBind-D -> e ┘
            -> 2-layer transformer over the present sensor tokens + [CLS]
            -> heads: deficit (binary) · stage (ordinal, cumulative logits) · turgor deviation

- Missing sensors are masked out of attention, so any subset of sensors works.
- Modality dropout (p = 0.3) removes whole sensors during training; at least one stays.
- z(.) standardises each embedding feature with means and spreads fitted on the
  fold's training plants only (`fit_standardiser`, rule 6). A per-sample layer
  norm would erase offsets shared by all features, so it is not used here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from uiwue.fusion.labels import N_STAGES


class FusionModel(nn.Module):
    def __init__(
        self,
        dims: dict[str, int],
        d_model: int = 256,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        modality_dropout: float = 0.3,
        n_stages: int = N_STAGES,
    ):
        super().__init__()
        self.sensors = list(dims)
        self.modality_dropout = modality_dropout
        for s, d in dims.items():  # identity until fit_standardiser is called
            self.register_buffer(f"mu_{s}", torch.zeros(2, d))
            self.register_buffer(f"sd_{s}", torch.ones(2, d))
        self.proj = nn.ModuleDict({s: nn.Linear(2 * d, d_model) for s, d in dims.items()})
        self.sensor_token = nn.Parameter(torch.randn(len(dims), d_model) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=2 * d_model, dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head_deficit = nn.Linear(d_model, 1)
        self.head_stage = nn.Linear(d_model, n_stages - 1)
        self.head_turgor = nn.Linear(d_model, 1)

    @torch.no_grad()
    def fit_standardiser(self, sensor: str, now: torch.Tensor, delta: torch.Tensor) -> None:
        """Per-feature mean and spread of e(t) and e(t) - e(baseline), from training rows only."""
        for i, x in enumerate((now, delta)):
            getattr(self, f"mu_{sensor}")[i] = x.mean(0)
            getattr(self, f"sd_{sensor}")[i] = x.std(0).clamp_min(1e-3) if len(x) > 1 else 1.0

    def _drop_modalities(self, present: torch.Tensor) -> torch.Tensor:
        if not self.training or self.modality_dropout <= 0:
            return present
        keep = present & (torch.rand_like(present, dtype=torch.float) >= self.modality_dropout)
        empty = ~keep.any(1) & present.any(1)
        if empty.any():  # restore one present sensor at random
            score = torch.rand_like(present, dtype=torch.float).masked_fill(~present, -1)
            pick = F.one_hot(score.argmax(1), present.shape[1]).bool()
            keep = torch.where(empty[:, None], pick, keep)
        return keep

    def forward(self, now: dict[str, torch.Tensor], base: dict[str, torch.Tensor], present: torch.Tensor) -> dict:
        """now/base: sensor -> (B, dim); present: (B, n_sensors) bool, in self.sensors order."""
        present = self._drop_modalities(present)
        tokens = []
        for i, s in enumerate(self.sensors):
            e, b = now[s], base[s]
            mu, sd = getattr(self, f"mu_{s}"), getattr(self, f"sd_{s}")
            t = self.proj[s](torch.cat([(e - mu[0]) / sd[0], (e - b - mu[1]) / sd[1]], -1))
            tokens.append(t + self.sensor_token[i])
        x = torch.cat([self.cls.expand(present.shape[0], -1, -1), torch.stack(tokens, 1)], 1)
        pad = torch.cat([torch.zeros_like(present[:, :1]), ~present], 1)
        z = self.norm(self.transformer(x, src_key_padding_mask=pad)[:, 0])
        return dict(
            logit=self.head_deficit(z).squeeze(-1),
            stage_logits=self.head_stage(z),
            turgor=self.head_turgor(z).squeeze(-1),
        )


def expected_stage(stage_logits: torch.Tensor) -> torch.Tensor:
    """Cumulative logits P(stage > k) -> expected stage in 0..n_stages-1."""
    return torch.sigmoid(stage_logits).sum(-1)


def fusion_loss(out: dict, y: torch.Tensor, stage: torch.Tensor, turgor: torch.Tensor, w_stage=0.5, w_turgor=0.5):
    """Deficit BCE + ordinal stage BCE (clip plants) + Huber on scaled turgor (clip runs)."""
    loss = F.binary_cross_entropy_with_logits(out["logit"], y.float())
    has_stage = stage >= 0
    if has_stage.any():
        k = torch.arange(out["stage_logits"].shape[1], device=y.device)
        target = (stage[has_stage, None] > k).float()
        loss = loss + w_stage * F.binary_cross_entropy_with_logits(out["stage_logits"][has_stage], target)
    has_t = torch.isfinite(turgor)
    if has_t.any():
        loss = loss + w_turgor * F.smooth_l1_loss(out["turgor"][has_t], turgor[has_t])
    return loss
