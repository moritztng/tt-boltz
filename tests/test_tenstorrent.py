import pytest
import torch
import os
from pathlib import Path
from functools import partial

from tt_boltz.tenstorrent import filter_dict, PairformerModule, MSAModule, DiffusionModule
from tt_boltz.reference import MSAModule as MSAModuleTorch, DiffusionModule as DiffusionModuleTorch
from tt_boltz.reference import PairformerModule as PairformerModuleTorch, PairformerNoSeqModule as PairformerNoSeqModuleTorch
from tt_boltz.boltz2 import get_indexing_matrix, single_to_keys

torch.set_grad_enabled(False)
torch.manual_seed(893)

CACHE = Path(os.environ.get("BOLTZ_CACHE", "~/.boltz")).expanduser()
STATE = torch.load(CACHE / "boltz2_conf.ckpt", map_location="cpu", mmap=True, weights_only=False)["state_dict"]
STATE_AFF = torch.load(CACHE / "boltz2_aff.ckpt", map_location="cpu", mmap=True, weights_only=False)["state_dict"]


def load(tt, ref, state, key, strict=False):
    sd = filter_dict(state, key)
    tt.load_state_dict(sd, strict=strict)
    ref.load_state_dict(sd, strict=strict)


def check(a, b, tol=0.1):
    err = ((a - b).abs() / b.abs()).median().item()
    assert err < tol, f"error {err:.4e} >= {tol}"


@pytest.mark.parametrize("seq_len", [100, 500])
def test_pairformer(seq_len):
    tt, ref = PairformerModule(2, 32, 4, 24, 16, transform_s=True), PairformerModuleTorch(384, 128, 2, v2=True).eval()
    load(tt, ref, STATE, "pairformer_module")

    s = 8 * torch.randn(1, seq_len, 384)
    z = 26 * torch.randn(1, seq_len, seq_len, 128)
    mask, pair_mask = torch.ones(1, seq_len), torch.ones(1, seq_len, seq_len)

    s_tt, z_tt = tt(s, z)
    s_ref, z_ref = ref(s, z, mask, pair_mask)
    check(s_tt, s_ref)
    check(z_tt, z_ref)


@pytest.mark.parametrize("seq_len", [100, 500])
def test_template_pairformer(seq_len):
    tt, ref = PairformerModule(2, 32, 4, None, None, transform_s=False), PairformerNoSeqModuleTorch(64, 2).eval()
    load(tt, ref, STATE, "template_module.pairformer")

    z = 26 * torch.randn(1, seq_len, seq_len, 64)
    check(tt(None, z)[1], ref(z, torch.ones(1, seq_len, seq_len)))


@pytest.mark.parametrize("seq_len", [100, 500])
def test_affinity_pairformer(seq_len):
    tt, ref = PairformerModule(4, 32, 4, None, None, transform_s=False, affinity=True), PairformerNoSeqModuleTorch(128, 4, v2=True).eval()
    load(tt, ref, STATE_AFF, "affinity_module1.pairformer_stack")

    z = 26 * torch.randn(1, seq_len, seq_len, 128)
    mask = torch.ones(1, seq_len)
    mask[0, :seq_len // 2] = 0
    mask = mask[:, torch.randperm(seq_len)]
    pair_mask = mask[:, :, None] * mask[:, None, :]

    check(tt(None, z, pair_mask=pair_mask)[1], ref(z, pair_mask=pair_mask))


@pytest.mark.parametrize("n_tokens,n_atoms,n_pairs", [(117, 928, 29), (574, 4384, 137)])
@pytest.mark.parametrize("n_samples", [1, 2])
def test_diffusion(n_tokens, n_atoms, n_pairs, n_samples):
    tt, ref = DiffusionModule(), DiffusionModuleTorch(384, 128, token_transformer_heads=16).eval()
    load(tt, ref, STATE, "structure_module.score_model")

    r_noisy = torch.randn(n_samples, n_atoms, 3)
    times = torch.randn(n_samples)
    s_inputs = torch.randn(1, n_tokens, 384)
    s_trunk = torch.randn(1, n_tokens, 384)
    q, c = torch.randn(1, n_atoms, 128), torch.randn(1, n_atoms, 128)
    bias_encoder = torch.randn(1, n_pairs, 32, 128, 12)
    bias_decoder = torch.randn(1, n_pairs, 32, 128, 12)
    bias_token = torch.randn(1, n_tokens, n_tokens, 384)
    keys = get_indexing_matrix(n_pairs, 32, 128, "cpu")

    r_tt = tt(r_noisy, times, s_inputs, s_trunk, q, c, bias_encoder, bias_token, bias_decoder, keys,
              torch.ones(1, n_atoms), torch.ones(1, n_atoms, n_tokens))
    r_ref = ref(
        r_noisy=r_noisy, times=times, s_inputs=s_inputs, s_trunk=s_trunk,
        diffusion_conditioning={
            "q": q, "c": c,
            "atom_enc_bias": bias_encoder, "token_trans_bias": bias_token, "atom_dec_bias": bias_decoder,
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
    check(r_tt, r_ref, tol=0.12)


@pytest.mark.parametrize("seq_len", [100, 500, 1000])
@pytest.mark.parametrize("n_sequences", [100])
def test_msa(seq_len, n_sequences):

    tt, ref = MSAModule(4, 32, 8, 32, 4), MSAModuleTorch(64, 128, 384, 4, 0, 0).eval()
    load(tt, ref, STATE, "msa_module", strict=True)

    z = 7 * torch.randn(1, seq_len, seq_len, 128)
    emb = torch.ones(1, seq_len, 384)
    feats = {
        "msa": torch.randint(33, (1, n_sequences, seq_len)),
        "has_deletion": torch.zeros(1, n_sequences, seq_len, dtype=torch.bool),
        "deletion_value": torch.zeros(1, n_sequences, seq_len),
        "msa_paired": torch.zeros(1, n_sequences, seq_len),
        "msa_mask": torch.ones(1, n_sequences, seq_len),
        "token_pad_mask": torch.ones(1, seq_len),
    }

    check(tt(z, emb, feats), ref(z, emb, feats))
