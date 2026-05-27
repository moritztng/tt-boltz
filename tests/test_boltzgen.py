"""Verify tt-boltz's ttnn modules are numerically equivalent to BoltzGen's
PyTorch reference modules on random weights.

If a test here passes, the corresponding tt-boltz ttnn module can be dropped
into BoltzGen's Boltz class as a same-state_dict replacement.
"""
import os
from functools import partial
from pathlib import Path

import pytest
import torch

from tt_boltz.tenstorrent import (
    DiffusionModule,
    MiniformerModule,
    MSAModule,
    PairformerModule,
)
from tt_boltz.boltz2 import get_indexing_matrix, single_to_keys


# Real-checkpoint tests are gated on the presence of the BoltzGen folding
# checkpoint (~2 GB). Set BOLTZGEN_FOLD_CKPT to point at boltz2_conf_final.ckpt
# to enable them.
_DEFAULT_CKPT = (
    Path.home()
    / ".cache/huggingface/hub/models--boltzgen--boltzgen-1"
    / "snapshots/c1be29e1f82ffcc72264f64b993c43fb4e0d17f0/boltz2_conf_final.ckpt"
)
BOLTZGEN_FOLD_CKPT = Path(os.environ.get("BOLTZGEN_FOLD_CKPT", _DEFAULT_CKPT))
requires_fold_ckpt = pytest.mark.skipif(
    not BOLTZGEN_FOLD_CKPT.exists(),
    reason=f"BoltzGen folding checkpoint not found at {BOLTZGEN_FOLD_CKPT}",
)

# Imported lazily inside tests so collection works even if a sibling import breaks.


torch.set_grad_enabled(False)
torch.manual_seed(893)


def _median_rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).abs() / b.abs().clamp_min(1e-6)).median().item()


def check(a: torch.Tensor, b: torch.Tensor, tol: float = 0.1) -> None:
    err = _median_rel_err(a, b)
    assert err < tol, f"median rel error {err:.4e} >= {tol}"


@pytest.mark.parametrize("seq_len", [100])
def test_pairformer_matches_boltzgen_reference(seq_len: int) -> None:
    """tt-boltz ttnn PairformerModule vs BoltzGen's reference PairformerModule.

    Both are constructed with the Boltz-2 folding-checkpoint shape (token_s=384,
    token_z=128, 2 blocks for speed, num_heads=16 → att_head_dim=24, pairwise
    head_width=32, pairwise_num_heads=4). Random-init the BoltzGen reference;
    load its state_dict into the ttnn module; compare forward outputs.
    """
    from boltzgen.model.layers.pairformer import PairformerModule as BGPairformer

    ref = BGPairformer(
        token_s=384,
        token_z=128,
        num_blocks=2,
        num_heads=16,
        dropout=0.0,
        pairwise_head_width=32,
        pairwise_num_heads=4,
    ).eval()

    tt = PairformerModule(
        n_blocks=2,
        tri_att_head_dim=32,
        tri_att_n_heads=4,
        att_head_dim=24,
        att_n_heads=16,
        transform_s=True,
    )

    sd = ref.state_dict()
    tt.load_state_dict(sd, strict=False)

    s = 8 * torch.randn(1, seq_len, 384)
    z = 26 * torch.randn(1, seq_len, seq_len, 128)
    mask = torch.ones(1, seq_len)
    pair_mask = torch.ones(1, seq_len, seq_len)

    s_tt, z_tt = tt(s, z, mask=mask, pair_mask=pair_mask)
    s_ref, z_ref = ref(s, z, mask, pair_mask)

    check(s_tt, s_ref)
    check(z_tt, z_ref)


@pytest.mark.parametrize("seq_len", [100])
def test_miniformer_matches_boltzgen_reference(seq_len: int) -> None:
    """tt-boltz ttnn MiniformerModule vs BoltzGen's reference MiniformerModule.

    Miniformer is BoltzGen's design-stage pairformer variant: replaces the 4
    triangular ops (out/in mul, start/end attention) with a single
    MiniTriangularUpdate that does a fused bi-directional update at D/4 width.
    The rest of the layer (pre_norm_s, attention, transitions, s_post_norm)
    matches Pairformer exactly.
    """
    from boltzgen.model.layers.miniformer import MiniformerModule as BGMiniformer

    ref = BGMiniformer(
        token_s=384, token_z=128, num_blocks=2, num_heads=16,
        dropout=0.0, post_layer_norm=False, activation_checkpointing=False,
    ).eval()

    tt = MiniformerModule(
        n_blocks=2,
        att_head_dim=24,  # token_s // num_heads
        att_n_heads=16,
    )

    sd = ref.state_dict()
    tt.load_state_dict(sd, strict=False)

    s = 8 * torch.randn(1, seq_len, 384)
    z = 26 * torch.randn(1, seq_len, seq_len, 128)
    mask = torch.ones(1, seq_len)
    pair_mask = torch.ones(1, seq_len, seq_len)

    s_tt, z_tt = tt(s, z, mask=mask, pair_mask=pair_mask)
    s_ref, z_ref = ref(s, z, mask, pair_mask)

    check(s_tt, s_ref)
    check(z_tt, z_ref)


@pytest.mark.parametrize("seq_len,n_sequences", [(100, 64)])
def test_msa_matches_boltzgen_reference(seq_len: int, n_sequences: int) -> None:
    """tt-boltz ttnn MSAModule vs BoltzGen reference MSAModule (folding-mode,
    miniformer_blocks=False — what the Boltz-2 folding checkpoint uses)."""
    from boltzgen.model.modules.trunk import MSAModule as BGMSAModule
    from boltzgen.data import const as bg_const

    ref = BGMSAModule(
        msa_s=64,
        token_z=128,
        token_s=384,
        msa_blocks=2,
        msa_dropout=0.0,
        z_dropout=0.0,
        miniformer_blocks=False,
        pairwise_head_width=32,
        pairwise_num_heads=4,
        use_paired_feature=True,  # matches all trained BoltzGen folding configs
    ).eval()

    tt = MSAModule(
        n_blocks=2,
        avg_head_dim=32,
        avg_n_heads=8,
        tri_att_head_dim=32,
        tri_att_n_heads=4,
    )

    sd = ref.state_dict()
    tt.load_state_dict(sd, strict=False)

    z = 7 * torch.randn(1, seq_len, seq_len, 128)
    emb = torch.ones(1, seq_len, 384)
    feats = {
        "msa": torch.randint(bg_const.num_tokens, (1, n_sequences, seq_len)),
        "has_deletion": torch.zeros(1, n_sequences, seq_len, dtype=torch.bool),
        "deletion_value": torch.zeros(1, n_sequences, seq_len),
        "msa_paired": torch.zeros(1, n_sequences, seq_len),
        "msa_mask": torch.ones(1, n_sequences, seq_len),
        "token_pad_mask": torch.ones(1, seq_len),
    }

    z_tt = tt(z, emb, feats)
    z_ref = ref(z, emb, feats)
    check(z_tt, z_ref)


@pytest.mark.parametrize("n_tokens,n_atoms,n_pairs", [(117, 928, 29)])
@pytest.mark.parametrize("n_samples", [1])
def test_diffusion_matches_boltzgen_reference(
    n_tokens: int, n_atoms: int, n_pairs: int, n_samples: int
) -> None:
    """tt-boltz ttnn DiffusionModule vs BoltzGen inner DiffusionModule
    (score_model) configured for folding mode."""
    from boltzgen.model.modules.diffusion import DiffusionModule as BGDiffusion

    ref = BGDiffusion(
        token_s=384,
        atom_s=128,
        atoms_per_window_queries=32,
        atoms_per_window_keys=128,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        token_transformer_depth=24,  # tt-boltz Diffusion hardcodes 24 layers (TOKEN_N_LAYERS)
        token_transformer_heads=16,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
        use_miniformer=False,
        gaussian_random_3d_encoding_dim=0,
        transformer_post_ln=False,
        predict_res_type=False,
        use_qk_norm=False,
    ).eval()

    tt = DiffusionModule()
    sd = ref.state_dict()
    tt.load_state_dict(sd, strict=False)

    r_noisy = torch.randn(n_samples, n_atoms, 3)
    times = torch.randn(n_samples)
    s_inputs = torch.randn(1, n_tokens, 384)
    s_trunk = torch.randn(1, n_tokens, 384)
    q = torch.randn(1, n_atoms, 128)
    c = torch.randn(1, n_atoms, 128)
    bias_encoder = torch.randn(1, n_pairs, 32, 128, 12)
    bias_decoder = torch.randn(1, n_pairs, 32, 128, 12)
    bias_token = torch.randn(1, n_tokens, n_tokens, 384)
    keys = get_indexing_matrix(n_pairs, 32, 128, "cpu")

    r_tt = tt(
        r_noisy, times, s_inputs, s_trunk, q, c,
        bias_encoder, bias_token, bias_decoder, keys,
        torch.ones(1, n_atoms), torch.ones(1, n_atoms, n_tokens),
    )
    out = ref(
        r_noisy=r_noisy, times=times, s_inputs=s_inputs, s_trunk=s_trunk,
        diffusion_conditioning={
            "q": q, "c": c,
            "atom_enc_bias": bias_encoder,
            "token_trans_bias": bias_token,
            "atom_dec_bias": bias_decoder,
            "to_keys": partial(single_to_keys, indexing_matrix=keys, W=32, H=128),
        },
        feats={
            "atom_pad_mask": torch.ones(1, n_atoms),
            "atom_to_token": torch.ones(1, n_atoms, n_tokens),
            "ref_pos": torch.randn(1, n_atoms, 3),
            "token_pad_mask": torch.ones(1, n_tokens),
        },
        multiplicity=n_samples,
    )
    check(r_tt, out["r_update"], tol=0.12)


def test_convert_to_tt_swap_round_trips() -> None:
    """convert_to_tt() swaps Pairformer/MSA/Diffusion in a Boltz-like container,
    survives a state_dict round-trip, and yields ttnn outputs that match the
    original PyTorch ones.

    Why a stub container: BoltzGen's full Boltz LightningModule takes ~30
    constructor args and pulls in trunk/template/affinity machinery that's
    irrelevant to the swap. The swap only cares about the attribute names
    ``pairformer_module`` / ``msa_module`` / ``structure_module.score_model``.
    """
    from boltzgen.model.layers.pairformer import PairformerModule as BGPairformer
    from boltzgen.model.modules.trunk import MSAModule as BGMSAModule
    from boltzgen.model.modules.diffusion import (
        AtomDiffusion as BGAtomDiffusion,
    )
    from boltzgen.data import const as bg_const

    from tt_boltz.boltzgen import convert_to_tt

    class StubBoltz(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pairformer_module = BGPairformer(
                token_s=384, token_z=128, num_blocks=2, num_heads=16,
                dropout=0.0, pairwise_head_width=32, pairwise_num_heads=4,
            )
            self.msa_module = BGMSAModule(
                msa_s=64, token_z=128, token_s=384, msa_blocks=2,
                msa_dropout=0.0, z_dropout=0.0,
                miniformer_blocks=False, use_paired_feature=True,
            )
            self.structure_module = BGAtomDiffusion(
                score_model_args=dict(
                    token_s=384, atom_s=128,
                    atoms_per_window_queries=32, atoms_per_window_keys=128,
                    atom_encoder_depth=3, atom_encoder_heads=4,
                    token_transformer_depth=24, token_transformer_heads=16,
                    atom_decoder_depth=3, atom_decoder_heads=4,
                    use_miniformer=False, gaussian_random_3d_encoding_dim=0,
                    transformer_post_ln=False, predict_res_type=False,
                    use_qk_norm=False,
                ),
            )

    ref = StubBoltz().eval()
    saved = {k: v.clone() for k, v in ref.state_dict().items()}

    swapped = StubBoltz().eval()
    convert_to_tt(swapped)
    # After convert, state_dict still loads from the saved PyTorch weights.
    swapped.load_state_dict(saved, strict=False)

    # Pairformer round-trip
    seq_len = 100
    s = 8 * torch.randn(1, seq_len, 384)
    z = 26 * torch.randn(1, seq_len, seq_len, 128)
    mask = torch.ones(1, seq_len)
    pair_mask = torch.ones(1, seq_len, seq_len)
    s_ref, z_ref = ref.pairformer_module(s, z, mask, pair_mask)
    s_tt, z_tt = swapped.pairformer_module(s, z, mask=mask, pair_mask=pair_mask)
    check(s_tt, s_ref)
    check(z_tt, z_ref)

    # MSA round-trip
    n_msa = 64
    z2 = 7 * torch.randn(1, seq_len, seq_len, 128)
    emb = torch.ones(1, seq_len, 384)
    feats = {
        "msa": torch.randint(bg_const.num_tokens, (1, n_msa, seq_len)),
        "has_deletion": torch.zeros(1, n_msa, seq_len, dtype=torch.bool),
        "deletion_value": torch.zeros(1, n_msa, seq_len),
        "msa_paired": torch.zeros(1, n_msa, seq_len),
        "msa_mask": torch.ones(1, n_msa, seq_len),
        "token_pad_mask": torch.ones(1, seq_len),
    }
    z_msa_ref = ref.msa_module(z2, emb, feats)
    z_msa_tt = swapped.msa_module(z2, emb, feats)
    check(z_msa_tt, z_msa_ref)


@pytest.mark.parametrize("seq_len", [100])
def test_pairformer_noseq_matches_boltzgen_reference(seq_len: int) -> None:
    """tt-boltz's no-seq Pairformer adapter vs BoltzGen's PairformerNoSeqModule.
    This is the variant inside BoltzGen's TemplateModule.pairformer."""
    from boltzgen.model.layers.pairformer import (
        PairformerNoSeqModule as BGPairformerNoSeq,
    )

    from tt_boltz.boltzgen import TTPairformerNoSeqModule

    ref = BGPairformerNoSeq(
        token_z=64, num_blocks=2,
        dropout=0.0, pairwise_head_width=32, pairwise_num_heads=4,
    ).eval()

    tt = TTPairformerNoSeqModule(n_blocks=2)
    tt.load_state_dict(ref.state_dict(), strict=False)

    z = 26 * torch.randn(1, seq_len, seq_len, 64)
    pair_mask = torch.ones(1, seq_len, seq_len)

    z_ref = ref(z, pair_mask)
    z_tt = tt(z, pair_mask)
    check(z_tt, z_ref)


def test_convert_to_tt_swaps_miniformer() -> None:
    """convert_to_tt routes a Miniformer-based Boltz to the ttnn Miniformer.
    Verifies the duck-typed dispatch in convert_to_tt and the state_dict
    flow into the Miniformer wrapper."""
    from boltzgen.model.layers.miniformer import MiniformerModule as BGMiniformer

    from tt_boltz.boltzgen import convert_to_tt
    from tt_boltz.tenstorrent import MiniformerModule as TTMiniformer

    class StubDesign(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pairformer_module = BGMiniformer(
                token_s=384, token_z=128, num_blocks=2, num_heads=16,
                dropout=0.0,
            )

    ref = StubDesign().eval()
    saved = {k: v.clone() for k, v in ref.state_dict().items()}

    swapped = StubDesign().eval()
    convert_to_tt(swapped)
    assert isinstance(swapped.pairformer_module, TTMiniformer)
    swapped.load_state_dict(saved, strict=False)

    seq_len = 100
    s = 8 * torch.randn(1, seq_len, 384)
    z = 26 * torch.randn(1, seq_len, seq_len, 128)
    mask = torch.ones(1, seq_len)
    pair_mask = torch.ones(1, seq_len, seq_len)

    s_ref, z_ref = ref.pairformer_module(s, z, mask, pair_mask)
    s_tt, z_tt = swapped.pairformer_module(s, z, mask=mask, pair_mask=pair_mask)
    check(s_tt, s_ref)
    check(z_tt, z_ref)


def _load_real_boltz():
    """Build a BoltzGen ``Boltz`` model from the real folding checkpoint.

    Returns a fresh model each call — call twice when you need both a PyTorch
    reference and a separately-swapped ttnn version, since convert_to_tt is
    destructive.
    """
    import inspect
    from boltzgen.model.models.boltz import Boltz

    ckpt = torch.load(
        BOLTZGEN_FOLD_CKPT, map_location="cpu", weights_only=False, mmap=True
    )
    sig = inspect.signature(Boltz.__init__).parameters
    hp = {k: v for k, v in ckpt["hyper_parameters"].items() if k in sig}
    model = Boltz(**hp).eval()
    return model, ckpt["state_dict"]


@requires_fold_ckpt
def test_real_checkpoint_loads_after_swap() -> None:
    """boltz2_conf_final.ckpt loads cleanly after convert_to_tt — zero missing,
    zero unexpected keys. This is the production state_dict seam test."""
    from tt_boltz.boltzgen import convert_to_tt

    model, state_dict = _load_real_boltz()
    convert_to_tt(model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Both should be zero; unrelated affinity/training keys were filtered above.
    assert len(missing) == 0, f"missing keys after swap: {missing[:5]}"
    assert len(unexpected) == 0, f"unexpected keys after swap: {unexpected[:5]}"
    # Confirm the swap actually happened.
    from tt_boltz.tenstorrent import (
        PairformerModule as TTP,
        MSAModule as TTM,
        DiffusionModule as TTD,
    )
    assert isinstance(model.pairformer_module, TTP)
    assert isinstance(model.msa_module, TTM)
    assert isinstance(model.structure_module.score_model, TTD)


@requires_fold_ckpt
def test_real_checkpoint_pairformer_numerical() -> None:
    """Run swapped Pairformer with the real checkpoint weights and compare
    against the pure-PyTorch reference. Loose tolerance because production
    Pairformer is 48 blocks deep and bfloat16 error accumulates."""
    from tt_boltz.boltzgen import convert_to_tt

    ref, state_dict = _load_real_boltz()
    ref.load_state_dict(state_dict, strict=False)

    tt, _ = _load_real_boltz()
    convert_to_tt(tt)
    tt.load_state_dict(state_dict, strict=False)

    seq_len = 100
    s = 8 * torch.randn(1, seq_len, 384)
    z = 26 * torch.randn(1, seq_len, seq_len, 128)
    mask = torch.ones(1, seq_len)
    pair_mask = torch.ones(1, seq_len, seq_len)

    with torch.no_grad():
        s_ref, z_ref = ref.pairformer_module(s, z, mask, pair_mask)
        s_tt, z_tt = tt.pairformer_module(s, z, mask=mask, pair_mask=pair_mask)

    # Production-depth (48-block) Pairformer accumulates more bfloat16 error
    # than the 2-block random-weight test. These tolerances reflect the
    # bfloat16-vs-fp32 floor of the ttnn implementation on Blackhole.
    check(s_tt, s_ref, tol=0.10)
    check(z_tt, z_ref, tol=0.30)
