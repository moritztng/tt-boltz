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
    """Replace BoltzGen's heavy PyTorch modules with tt-boltz ttnn modules in-place.

    Safe to call on any BoltzGen ``Boltz`` instance — modules that don't exist
    (e.g. ``msa_module`` is absent when ``inverse_fold=True``) are skipped.

    Must be called BEFORE ``load_state_dict``: the ttnn ``TorchWrapper`` captures
    its weights via ``_load_from_state_dict`` during the load.
    """
    pair = getattr(model, "pairformer_module", None)
    if pair is not None:
        n_blocks = pair.num_blocks
        n_heads = pair.num_heads
        att_head_dim = _att_head_dim(pair)
        if _is_miniformer(pair):
            model.pairformer_module = TTMiniformerModule(
                n_blocks=n_blocks,
                att_head_dim=att_head_dim,
                att_n_heads=n_heads,
            )
        else:
            model.pairformer_module = TTPairformerModule(
                n_blocks=n_blocks,
                tri_att_head_dim=_PAIRFORMER_TRI_HEAD_DIM,
                tri_att_n_heads=_PAIRFORMER_TRI_N_HEADS,
                att_head_dim=att_head_dim,
                att_n_heads=n_heads,
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
        structure.score_model = TTDiffusionModule()

    template = getattr(model, "template_module", None)
    inner = getattr(template, "pairformer", None) if template is not None else None
    # PairformerNoSeqLayer has tri_mul_out; MiniformerNoSeqLayer has triangular.
    # tt-boltz only ports the PairformerNoSeq variant today.
    if inner is not None and _is_pairformer_noseq(inner):
        template.pairformer = TTPairformerNoSeqModule(n_blocks=inner.num_blocks)

    return model


def _is_miniformer(pairformer: nn.Module) -> bool:
    """Detect a Miniformer-style layer (BoltzGen's design-stage variant)
    by the presence of a `triangular` attribute on a layer instead of the
    four `tri_*` ops Pairformer has."""
    layers = getattr(pairformer, "layers", None)
    if layers is None or len(layers) == 0:
        return False
    return hasattr(layers[0], "triangular")


def _has_miniformer_inner(msa: nn.Module) -> bool:
    """MSAModule uses MiniformerNoSeqLayer (design stage) when miniformer_blocks=True;
    tt-boltz's TTMSAModule only knows the PairformerNoSeqLayer variant, so we
    leave such modules on PyTorch."""
    layers = getattr(msa, "layers", None)
    if layers is None or len(layers) == 0:
        return False
    inner = getattr(layers[0], "pairformer_layer", None)
    return inner is not None and hasattr(inner, "triangular")


def _is_pairformer_noseq(module: nn.Module) -> bool:
    """PairformerNoSeqLayer has tri_mul_out; MiniformerNoSeqLayer has triangular."""
    layers = getattr(module, "layers", None)
    if layers is None or len(layers) == 0:
        return False
    return hasattr(layers[0], "tri_mul_out")


class TTPairformerNoSeqModule(TTPairformerModule):
    """No-seq Pairformer for BoltzGen's template_module.pairformer.

    Same ttnn pairformer stack as the trunk, but constructed with
    ``transform_s=False`` (skips the s-track) and a forward signature that
    matches ``PairformerNoSeqModule.forward(z, pair_mask, use_kernels=...)``.
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

    def forward(  # type: ignore[override]
        self, z, pair_mask, use_kernels: bool = False,
    ):
        _, z_out = super().forward(None, z, mask=None, pair_mask=pair_mask)
        return z_out


def _tt_load_from_checkpoint(checkpoint_path, *, strict=True, map_location="cpu",
                              weights_only=False, **kwargs):
    """Replacement for ``Boltz.load_from_checkpoint`` that:
      1. loads the checkpoint manually (avoiding Lightning's ``.to(device)``
         which trips on torchmetrics' CUDA dummy tensor when PyTorch is CPU-only),
      2. filters legacy hparams the current Boltz signature no longer accepts,
      3. applies ``convert_to_tt`` BEFORE ``load_state_dict`` so the ttnn
         wrappers capture their weights via ``_load_from_state_dict``.
    """
    import inspect

    import torch as _torch
    from boltzgen.model.models.boltz import Boltz

    ckpt = _torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    sig = inspect.signature(Boltz.__init__).parameters
    hp = {k: v for k, v in ckpt["hyper_parameters"].items() if k in sig}
    hp.update({k: v for k, v in kwargs.items() if k in sig})
    model = Boltz(**hp)
    convert_to_tt(model)
    state = _remap_legacy_state_dict_keys(ckpt["state_dict"])
    model.load_state_dict(state, strict=False)
    return model.eval()


def _remap_legacy_state_dict_keys(state: dict) -> dict:
    """Mirror ``Boltz.on_load_checkpoint``'s key remap, which Lightning's
    ``load_from_checkpoint`` would normally apply. Older checkpoints wrapped
    the diffusion token transformer in a ``token_transformer_layers`` ModuleList;
    current Boltz code stores it as ``token_transformer`` directly. The
    BoltzGen design checkpoint (boltzgen1_diverse.ckpt) is the older format."""
    return {
        k.replace(".token_transformer_layers.0.", ".token_transformer."): v
        for k, v in state.items()
    }


def cli_main() -> None:
    """Entry point: run ``boltzgen <args>`` with ``Boltz.load_from_checkpoint``
    monkey-patched to apply convert_to_tt, and ``--no_subprocess`` forced
    so the patch carries through every pipeline step.

    Use exactly like ``boltzgen``:
        tt-boltzgen run example/prot.yaml --output out/ ...
    """
    import sys

    from boltzgen.cli.boltzgen import main as _bg_main
    from boltzgen.model.models.boltz import Boltz

    Boltz.load_from_checkpoint = _tt_load_from_checkpoint  # type: ignore[assignment]

    args = sys.argv[1:]
    if args and args[0] == "run" and "--no_subprocess" not in args:
        args.insert(1, "--no_subprocess")
    sys.argv = [sys.argv[0]] + args
    _bg_main()


if __name__ == "__main__":
    cli_main()


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
