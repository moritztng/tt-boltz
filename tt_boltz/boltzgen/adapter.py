"""Swap BoltzGen's heavy PyTorch modules for tt-boltz's ttnn equivalents.

The vendored BoltzGen model classes (Pairformer / MSA / DiffusionModule /
template Pairformer) have the same state_dict layout as tt-boltz's ttnn
equivalents in :mod:`tt_boltz.tenstorrent`. ``convert_to_tt`` replaces the
matching attributes on a constructed model in place; weights then flow into
ttnn during the standard ``load_state_dict`` call (TorchWrapper intercepts
``_load_from_state_dict``).

Two small calling-convention adapters live here too:

* :class:`TTScoreModelAdapter` — BoltzGen calls the diffusion score model with
  kwargs (`r_noisy=`, `feats=`, `diffusion_conditioning=`, …) and reads
  ``net_out["r_update"]``. tt-boltz's :class:`TTDiffusionModule` takes
  positional args and returns the raw tensor.

* :class:`TTPairformerNoSeqModule` — BoltzGen's TemplateModule.pairformer
  ignores the sequence track; we construct the underlying ttnn Pairformer
  with ``transform_s=False`` and adapt ``forward(z, pair_mask)``.

The legacy state-dict key remap (``token_transformer_layers.0.*`` →
``token_transformer.*``) is also here, because some older shipping
checkpoints predate BoltzGen's renaming and we want to load them cleanly.
"""
from __future__ import annotations

import inspect
import torch
import torch.nn as nn

from tt_boltz.tenstorrent import (
    DiffusionModule as TTDiffusionModule,
    MiniformerModule as TTMiniformerModule,
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
    """Replace BoltzGen's heavy PyTorch modules with tt-boltz ttnn modules in place.

    Idempotent: a second call on an already-converted model is a no-op.
    Safe to call on any BoltzGen ``Boltz`` instance — modules that don't exist
    (e.g. ``msa_module`` is absent when ``inverse_fold=True``) are skipped. Must
    be called BEFORE ``load_state_dict``: the ttnn ``TorchWrapper`` captures its
    weights via ``_load_from_state_dict`` during the load.
    """
    if getattr(model, "_tt_converted", False):
        return model
    pair = getattr(model, "pairformer_module", None)
    if pair is not None:
        att_head_dim = _att_head_dim(pair)
        if _is_miniformer(pair):
            model.pairformer_module = TTMiniformerModule(
                n_blocks=pair.num_blocks,
                att_head_dim=att_head_dim,
                att_n_heads=pair.num_heads,
            )
        else:
            model.pairformer_module = TTPairformerModule(
                n_blocks=pair.num_blocks,
                tri_att_head_dim=_PAIRFORMER_TRI_HEAD_DIM,
                tri_att_n_heads=_PAIRFORMER_TRI_N_HEADS,
                att_head_dim=att_head_dim,
                att_n_heads=pair.num_heads,
                transform_s=True,
            )

    msa = getattr(model, "msa_module", None)
    if msa is not None and not _has_miniformer_inner(msa):
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
        structure.score_model = TTScoreModelAdapter()

    template = getattr(model, "template_module", None)
    inner = getattr(template, "pairformer", None) if template is not None else None
    if inner is not None and _is_pairformer_noseq(inner):
        template.pairformer = TTPairformerNoSeqModule(n_blocks=inner.num_blocks)

    model._tt_converted = True
    return model


def load_boltz_checkpoint(
    checkpoint_path: str,
    *,
    strict: bool = False,
    map_location: str = "cpu",
    **constructor_overrides,
) -> nn.Module:
    """Build a ``Boltz`` model from a checkpoint with ``convert_to_tt`` applied.

    Filters legacy hyper_parameters the current Boltz signature no longer
    accepts, applies ``convert_to_tt`` before ``load_state_dict``, and remaps
    the legacy ``token_transformer_layers.0.*`` keys some older design
    checkpoints (e.g. ``boltzgen1_diverse.ckpt``) use.
    """
    from tt_boltz.boltzgen.model.models.boltz import Boltz

    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False, mmap=True)
    sig = inspect.signature(Boltz.__init__).parameters
    hp = {k: v for k, v in ckpt["hyper_parameters"].items() if k in sig}
    hp.update({k: v for k, v in constructor_overrides.items() if k in sig})

    model = Boltz(**hp)
    convert_to_tt(model)
    state = _remap_legacy_state_dict_keys(ckpt["state_dict"])
    model.load_state_dict(state, strict=strict)
    return model.eval()


class TTScoreModelAdapter(TTDiffusionModule):
    """Adapt the BoltzGen score-model calling convention to ``TTDiffusionModule``.

    BoltzGen's ``AtomDiffusion.preconditioned_network_forward`` calls
    ``score_model(r_noisy=..., times=..., s_inputs=..., s_trunk=...,
    feats=..., diffusion_conditioning=..., multiplicity=...)`` and reads
    ``net_out["r_update"]``. ``TTDiffusionModule`` takes positional args and
    returns the raw tensor, so this subclass shuffles kwargs into positional
    args, unwraps ``to_keys`` to its raw indexing matrix, and wraps the result
    in a dict. Inherits ``TorchWrapper``'s state_dict mechanism so weights
    still flow in during ``load_state_dict``.
    """

    def forward(  # type: ignore[override]
        self,
        *,
        r_noisy,
        times,
        s_inputs,
        s_trunk,
        feats,
        diffusion_conditioning,
        multiplicity: int = 1,
        **_unused,
    ):
        dc = diffusion_conditioning
        # ``to_keys`` is always ``partial(single_to_keys, indexing_matrix=K, W=32, H=128)``
        # built in encoders.py — unwrap the raw matrix.
        keys_indexing = dc["to_keys"].keywords["indexing_matrix"]
        r_update = super().forward(
            r_noisy, times, s_inputs, s_trunk,
            dc["q"], dc["c"],
            dc["atom_enc_bias"], dc["token_trans_bias"], dc["atom_dec_bias"],
            keys_indexing,
            feats["atom_pad_mask"], feats["atom_to_token"],
        )
        # BoltzGen reads only ["r_update"] from the design path; the other two
        # keys are unused (predict_res_type=False on every shipping ckpt).
        return {"r_update": r_update, "token_a": None, "res_type": None}


class TTPairformerNoSeqModule(TTPairformerModule):
    """No-seq Pairformer for BoltzGen's ``TemplateModule.pairformer``.

    Same ttnn pairformer stack as the trunk, but ``transform_s=False`` (no
    s-track) and a forward signature that matches
    ``PairformerNoSeqModule.forward(z, pair_mask, use_kernels=...)``.
    """

    def __init__(self, n_blocks: int):
        super().__init__(
            n_blocks=n_blocks,
            tri_att_head_dim=_PAIRFORMER_TRI_HEAD_DIM,
            tri_att_n_heads=_PAIRFORMER_TRI_N_HEADS,
            att_head_dim=None,
            att_n_heads=None,
            transform_s=False,
        )

    def forward(self, z, pair_mask, use_kernels: bool = False):  # type: ignore[override]
        _, z_out = super().forward(None, z, mask=None, pair_mask=pair_mask)
        return z_out


def _remap_legacy_state_dict_keys(state: dict) -> dict:
    """Map older ``token_transformer_layers.0.*`` keys to ``token_transformer.*``.

    Mirrors what BoltzGen's old ``Boltz.on_load_checkpoint`` did. The shipping
    design checkpoint (``boltzgen1_diverse.ckpt``) predates the rename.
    """
    return {
        k.replace(".token_transformer_layers.0.", ".token_transformer."): v
        for k, v in state.items()
    }


def _is_miniformer(pairformer: nn.Module) -> bool:
    """Miniformer layers have a single ``triangular`` op; Pairformer has four ``tri_*`` ops."""
    layers = getattr(pairformer, "layers", None)
    if layers is None or len(layers) == 0:
        return False
    return hasattr(layers[0], "triangular")


def _has_miniformer_inner(msa: nn.Module) -> bool:
    """True when MSALayer's inner pairformer is Miniformer (design ckpts only)."""
    layers = getattr(msa, "layers", None)
    if layers is None or len(layers) == 0:
        return False
    inner = getattr(layers[0], "pairformer_layer", None)
    return inner is not None and hasattr(inner, "triangular")


def _is_pairformer_noseq(module: nn.Module) -> bool:
    """PairformerNoSeqLayer has ``tri_mul_out``; MiniformerNoSeqLayer has ``triangular``."""
    layers = getattr(module, "layers", None)
    if layers is None or len(layers) == 0:
        return False
    return hasattr(layers[0], "tri_mul_out")


def _att_head_dim(pairformer: nn.Module) -> int:
    """Read the per-head dim of the layer's AttentionPairBias."""
    layer0 = pairformer.layers[0]
    attn = layer0.attention
    proj_q = getattr(attn, "proj_q", None)
    if proj_q is not None:
        return proj_q.out_features // attn.num_heads
    return layer0.pre_norm_s.normalized_shape[0] // pairformer.num_heads
