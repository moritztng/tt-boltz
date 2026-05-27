"""Verify tt-boltz's ttnn modules are numerically equivalent to BoltzGen's
PyTorch reference modules on random weights.

If a test here passes, the corresponding tt-boltz ttnn module can be dropped
into BoltzGen's Boltz class as a same-state_dict replacement.
"""
import pytest
import torch

from functools import partial

from tt_boltz.tenstorrent import PairformerModule, MSAModule, DiffusionModule
from tt_boltz.boltz2 import get_indexing_matrix, single_to_keys

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
