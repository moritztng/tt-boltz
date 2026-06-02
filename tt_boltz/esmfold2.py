"""ESMFold2 on Tenstorrent (ttnn) — folding trunk.

ESMFold2 is a diffusion all-atom structure predictor built on ESMC-6B
embeddings. Reference architecture: the Biohub fork of transformers
(models/esmfold2/modeling_esmfold2_common.py), see tests/esmfold2_reference.py.

This module implements the `folding_trunk`: 48 PairUpdateBlocks, each
    z = z + tri_mul_out(z); z = z + tri_mul_in(z); z = z + pair_transition(z)
operating on the pair representation z [B, L, L, c_z=256].

The triangle-multiplication math is identical to AlphaFold3 / Boltz, so we
reuse tt-boltz's proven ttnn `TriangleMultiplication` kernel and the ESMC
`SwiGLUFFN`, remapping ESMFold2's bundled weights into the expected layout:
  proj_bundle[:512]->p_in, proj_bundle[512:]->g_in, proj_emit->p_out,
  proj_gate->g_out, norm_start->norm_in, norm_mix->norm_out.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import ttnn

from tt_boltz.esmc import SwiGLUFFN
from tt_boltz.tenstorrent import (
    Module,
    TorchWrapper,
    TriangleMultiplication,
    Weights,
    WeightScope,
)

C_Z = 256  # pair channels
PAD_MULTIPLE = 32  # pad seq to a tile multiple so the triangle contraction excludes padding


def _remap_trimul(sd: dict, prefix: str) -> WeightScope:
    """ESMFold2 TriangleMultiplicativeBlock weights -> tt-boltz TriangleMultiplication."""
    g = lambda k: sd[f"{prefix}.{k}"]
    pb = g("proj_bundle.weight")  # [4*latent, c_z]; rows: signal(2*latent) | gate(2*latent)
    half = pb.shape[0] // 2
    return WeightScope({
        "norm_in.weight": g("norm_start.weight"),
        "norm_in.bias": g("norm_start.bias"),
        "norm_out.weight": g("norm_mix.weight"),
        "norm_out.bias": g("norm_mix.bias"),
        "p_in.weight": pb[:half],   # signal (left|right values)
        "g_in.weight": pb[half:],   # gate (left|right gates)
        "p_out.weight": g("proj_emit.weight"),
        "g_out.weight": g("proj_gate.weight"),
    })


def _remap_transition(sd: dict) -> WeightScope:
    """ESMFold2 pair_transition (LN + SwiGLUMLP w12/w3) -> ESMC SwiGLUFFN keys."""
    return WeightScope({
        "0.weight": sd["pair_transition.norm.weight"],
        "0.bias": sd["pair_transition.norm.bias"],
        "1.weight": sd["pair_transition.ffn.w12.weight"],
        "3.weight": sd["pair_transition.ffn.w3.weight"],
    })


class PairUpdateBlock(Module):
    """z = z + tri_mul_out(z); z = z + tri_mul_in(z); z = z + pair_transition(z)."""

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        sd = self.weights.as_dict()
        # tri_mul_out = outgoing (ending=False); tri_mul_in = incoming (ending=True).
        self.tri_out = TriangleMultiplication(
            False, _remap_trimul(sd, "tri_mul_out._engine"), compute_kernel_config
        )
        self.tri_in = TriangleMultiplication(
            True, _remap_trimul(sd, "tri_mul_in._engine"), compute_kernel_config
        )
        self.transition = SwiGLUFFN(_remap_transition(sd), compute_kernel_config)

    def _residual(self, z: ttnn.Tensor, update: ttnn.Tensor) -> ttnn.Tensor:
        out = ttnn.add(z, update)
        ttnn.deallocate(z)
        ttnn.deallocate(update)
        return out

    def __call__(self, z: ttnn.Tensor, mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        z = self._residual(z, self.tri_out(z, mask))
        z = self._residual(z, self.tri_in(z, mask))
        z = self._residual(z, self.transition(z))
        return z


class FoldingTrunkModel(Module):
    """ModuleList of PairUpdateBlocks over the pair representation z."""

    def __init__(self, n_layers: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.blocks = [
            PairUpdateBlock(self.scope(f"blocks.{i}"), compute_kernel_config)
            for i in range(n_layers)
        ]

    def __call__(self, z: ttnn.Tensor, mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        for block in self.blocks:
            z = block(z, mask)
        return z


class FoldingTrunk(TorchWrapper):
    """Top-level folding trunk (torch z [B,L,L,256] in/out)."""

    def __init__(self, n_layers: int = 48, d_pair: int = C_Z):
        super().__init__()
        self.n_layers = n_layers
        self.d_pair = d_pair

    def _create_module(self, weights: WeightScope) -> FoldingTrunkModel:
        return FoldingTrunkModel(self.n_layers, weights, self.compute_kernel_config)

    def forward(self, z):
        # Pad both sequence axes to a tile multiple and mask padding out of the
        # triangle contraction, then slice back (mirrors tt-boltz PairformerModule).
        seq_len = z.shape[1]
        pad = (-seq_len) % PAD_MULTIPLE
        mask = None
        if pad:
            z = F.pad(z, (0, 0, 0, pad, 0, pad))
            real = torch.zeros(1, z.shape[1], z.shape[1])
            real[:, :seq_len, :seq_len] = 1.0
            mask = self._from_torch(real)
        out = self.module(self._from_torch(z), mask)
        return self._to_torch(out)[:, :seq_len, :seq_len, :]
