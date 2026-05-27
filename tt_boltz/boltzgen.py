"""Adapter that swaps BoltzGen's PyTorch Pairformer/MSA/Diffusion modules
for tt-boltz's ttnn equivalents.

Usage:

    from boltzgen.model.models.boltz import Boltz
    from tt_boltz.boltzgen import convert_to_tt

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = Boltz(**ckpt["hyper_parameters"])
    convert_to_tt(model)            # swap modules BEFORE loading weights
    model.load_state_dict(ckpt["state_dict"])

The swap relies on identical state_dict key naming between BoltzGen and
tt-boltz reference modules — verified by tests/test_boltzgen.py.
"""
from __future__ import annotations

import torch.nn as nn

from tt_boltz.tenstorrent import (
    DiffusionModule as TTDiffusionModule,
    MSAModule as TTMSAModule,
    PairformerModule as TTPairformerModule,
)

# Constants baked into BoltzGen's MSALayer (PairWeightedAveraging c_h=32, num_heads=8).
_MSA_AVG_HEAD_DIM = 32
_MSA_AVG_N_HEADS = 8
# BoltzGen Pairformer default pairwise_head_width=32, pairwise_num_heads=4.
_PAIRFORMER_TRI_HEAD_DIM = 32
_PAIRFORMER_TRI_N_HEADS = 4


def convert_to_tt(model: nn.Module) -> nn.Module:
    """Replace BoltzGen's heavy PyTorch modules with tt-boltz ttnn modules in-place.

    Safe to call on any BoltzGen ``Boltz`` instance — modules that don't exist
    (e.g. ``msa_module`` is absent when ``inverse_fold=True``) are skipped.

    Must be called BEFORE ``load_state_dict``: the ttnn ``TorchWrapper`` captures
    its weights via ``_load_from_state_dict`` during the load.
    """
    pair = getattr(model, "pairformer_module", None)
    if pair is not None and not _is_miniformer(pair):
        n_blocks = pair.num_blocks
        n_heads = pair.num_heads
        # BoltzGen folding uses token_s=384; att_head_dim = token_s // num_heads.
        # We read num_heads off the existing module; head_dim is fixed at construction
        # time via token_s, so look it up on the first AttentionPairBias instance.
        att_head_dim = _att_head_dim(pair)
        model.pairformer_module = TTPairformerModule(
            n_blocks=n_blocks,
            tri_att_head_dim=_PAIRFORMER_TRI_HEAD_DIM,
            tri_att_n_heads=_PAIRFORMER_TRI_N_HEADS,
            att_head_dim=att_head_dim,
            att_n_heads=n_heads,
            transform_s=True,
        )

    msa = getattr(model, "msa_module", None)
    if msa is not None:
        model.msa_module = TTMSAModule(
            n_blocks=msa.msa_blocks,
            avg_head_dim=_MSA_AVG_HEAD_DIM,
            avg_n_heads=_MSA_AVG_N_HEADS,
            tri_att_head_dim=_PAIRFORMER_TRI_HEAD_DIM,
            tri_att_n_heads=_PAIRFORMER_TRI_N_HEADS,
        )

    structure = getattr(model, "structure_module", None)
    score = getattr(structure, "score_model", None) if structure is not None else None
    if score is not None:
        structure.score_model = TTDiffusionModule()

    return model


def _is_miniformer(pairformer: nn.Module) -> bool:
    """Detect a Miniformer-style layer (BoltzGen's design-stage variant)
    by the presence of a `triangular` attribute on a layer instead of the
    four `tri_*` ops Pairformer has."""
    layers = getattr(pairformer, "layers", None)
    if layers is None or len(layers) == 0:
        return False
    return hasattr(layers[0], "triangular")


def _att_head_dim(pairformer: nn.Module) -> int:
    """Read the AttentionPairBias head_dim off the first Pairformer layer."""
    layer0 = pairformer.layers[0]
    attn = layer0.attention
    # BoltzGen AttentionPairBias stores head_dim implicitly via dim_head; the
    # public API is the projection weight shape.
    proj_q = getattr(attn, "proj_q", None)
    if proj_q is not None:
        return proj_q.out_features // attn.num_heads
    # Fall back: token_s // num_heads where token_s is inferred from layer norm.
    return layer0.pre_norm_s.normalized_shape[0] // pairformer.num_heads
