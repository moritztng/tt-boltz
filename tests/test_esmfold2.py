"""Tests for the ttnn ESMFold2 implementation, against the Biohub-fork reference.

Same idiom as test_esmc.py: build the reference with random weights, load the
same state_dict into the ttnn module, compare on TT device 0.
"""

import os
import sys

import pytest
import torch
import ttnn

sys.path.insert(0, os.path.dirname(__file__))
from esmfold2_reference import (  # noqa: E402
    DIFFUSION_TOKEN,
    make_confidence_core,
    make_diffusion_conditioning,
    make_diffusion_module,
    make_diffusion_transformer,
    make_distogram_head,
    make_folding_trunk,
    make_inputs_embedder,
    make_msa_encoder,
    make_relpos,
    make_swa_atom_transformer,
)

from tt_boltz import esmfold2 as tt_ef2  # noqa: E402
from tt_boltz.tenstorrent import get_device  # noqa: E402

torch.set_grad_enabled(False)
torch.manual_seed(893)

C_Z = tt_ef2.C_Z


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.mark.parametrize("n_layers,seq_len", [(1, 32), (4, 64), (48, 48)])
def test_folding_trunk(n_layers, seq_len):
    ref = make_folding_trunk(n_layers=n_layers)
    z = torch.randn(1, seq_len, seq_len, C_Z)
    ref_out = ref(z)  # FoldingTrunk.forward(pair), mask=None

    mod = tt_ef2.FoldingTrunk(n_layers=n_layers)
    mod.load_state_dict(ref.state_dict(), strict=False)
    out = mod(z)

    assert out.shape == ref_out.shape, (out.shape, ref_out.shape)
    p = pcc(out, ref_out)
    assert p > 0.98, f"PCC {p:.5f} too low (n_layers={n_layers}, L={seq_len})"


@pytest.mark.parametrize("seq_len,depth", [(32, 8), (64, 16)])
def test_msa_encoder(seq_len, depth):
    ref = make_msa_encoder()
    L, M = seq_len, depth
    xp = torch.randn(1, L, L, 256)
    xi = torch.randn(1, L, 451)
    oh = torch.randn(1, L, M, 33)
    hd = torch.randn(1, L, M)
    dv = torch.randn(1, L, M)
    mm = torch.ones(1, L, M)
    ref_out = ref(xp, xi, oh, hd, dv, mm)

    mod = tt_ef2.MSAEncoder()
    mod.load_state_dict(ref.state_dict(), strict=False)
    out = mod(xp, xi, oh, hd, dv, mm)
    assert out.shape == ref_out.shape
    assert pcc(out, ref_out) > 0.99


@pytest.mark.parametrize("seq_len", [37, 64])
def test_relpos_encoding(seq_len):
    ref = make_relpos()
    L = seq_len
    residue_index = torch.arange(L).unsqueeze(0)
    asym_id = torch.zeros(1, L, dtype=torch.long)
    sym_id = torch.zeros(1, L, dtype=torch.long)
    entity_id = torch.zeros(1, L, dtype=torch.long)
    token_index = torch.arange(L).unsqueeze(0)
    ref_out = ref(residue_index, asym_id, sym_id, entity_id, token_index)

    mod = tt_ef2.RelPosEncoding(r_bins=32, c_bins=2)
    mod.load_state_dict(ref.state_dict(), strict=False)
    out = mod(residue_index, asym_id, sym_id, entity_id, token_index)
    assert out.shape == ref_out.shape
    assert pcc(out, ref_out) > 0.999


@pytest.mark.parametrize("n_tokens", [16, 32])
def test_inputs_embedder(n_tokens):
    ref = make_inputs_embedder()
    L = n_tokens
    apt = torch.randint(1, 6, (L,))
    tok_idx = torch.repeat_interleave(torch.arange(L), apt).unsqueeze(0)
    N = tok_idx.shape[1]
    aatype = torch.randn(1, L, 33)
    profile = torch.randn(1, L, 33)
    deletion_mean = torch.randn(1, L)
    ref_pos = torch.randn(1, N, 3)
    atom_mask = torch.ones(1, N)
    uid = torch.randint(0, 8, (1, N))
    ref_charge = torch.randn(1, N)
    ref_element = torch.randn(1, N, 128)
    ref_atom_name_chars = torch.randn(1, N, 4, 64)
    args = (aatype, profile, deletion_mean, ref_pos, atom_mask, uid, ref_charge,
            ref_element, ref_atom_name_chars, tok_idx)
    ref_out = ref(*args)

    mod = tt_ef2.InputsEmbedder(n_heads=4, n_blocks=3)
    mod.load_state_dict(ref.state_dict(), strict=False)
    out = mod(*args)
    assert out.shape == ref_out.shape == (1, L, 451)
    assert pcc(out, ref_out) > 0.98


@pytest.mark.parametrize("seq_len", [32, 64])
def test_distogram_head(seq_len):
    ref = make_distogram_head()
    z = torch.randn(1, seq_len, seq_len, 256)
    ref_out = ref(z + z.transpose(-2, -3))

    mod = tt_ef2.DistogramHeadModel()
    mod.load_state_dict(ref.state_dict(), strict=False)
    out = mod(z)
    assert out.shape == ref_out.shape
    assert pcc(out, ref_out) > 0.999


@pytest.mark.parametrize("seq_len", [32, 64])
def test_confidence_head(seq_len):
    ref = make_confidence_core()
    L = seq_len
    apt = torch.randint(1, 6, (L,))
    tok_idx = torch.repeat_interleave(torch.arange(L), apt).unsqueeze(0)
    intra = torch.cat([torch.arange(a) for a in apt]).unsqueeze(0)  # intra-token position
    A = tok_idx.shape[1]
    s_inputs = torch.randn(1, L, 451)
    z = torch.randn(1, L, L, 256)
    rep_coords = torch.randn(1, L, 3) * 10
    ref_out = ref(s_inputs, z, rep_coords, tok_idx, intra)

    mod = tt_ef2.ConfidenceHead(conf_trunk_layers=4)
    mod.load_state_dict(ref.state_dict(), strict=False)
    out = mod(s_inputs, z, rep_coords, tok_idx, intra)

    names = ["pae", "pde", "plddt", "resolved"]
    for name, o, r in zip(names, out, ref_out):
        assert o.shape == r.shape, (name, o.shape, r.shape)
        p = pcc(o, r)
        assert p > 0.98, f"{name} PCC {p:.5f} too low (L={seq_len})"


@pytest.mark.parametrize("n_tokens", [8, 16])
def test_diffusion_module(n_tokens):
    torch.manual_seed(0)
    ref = make_diffusion_module()
    L = n_tokens
    apt = torch.randint(1, 8, (L,))  # atoms per token
    tok_idx = torch.repeat_interleave(torch.arange(L), apt).unsqueeze(0)  # [1,N]
    N = tok_idx.shape[1]

    x_noisy = torch.randn(1, N, 3)
    ref_pos = torch.randn(1, N, 3)
    ref_charge = torch.randn(1, N)
    ref_mask = torch.ones(1, N)
    ref_element = torch.randn(1, N, 128)
    ref_atom_name_chars = torch.randn(1, N, 4, 64)
    ref_space_uid = torch.randint(0, 8, (1, N))
    s_inputs = torch.randn(1, L, 451)
    z_trunk = torch.randn(1, L, L, 256)
    relpos = torch.randn(1, L, L, 256)
    t_hat = torch.tensor([12.0])
    zL = torch.zeros(1, L, dtype=torch.long)

    ref_out = ref(
        x_noisy, t_hat, ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars,
        ref_space_uid, tok_idx, s_inputs, None, z_trunk, relpos, zL, zL, zL, zL, zL,
    )["x_denoised"]

    mod = tt_ef2.DiffusionModule(sigma_data=16.0)
    mod.load_state_dict(ref.state_dict(), strict=False)
    out = mod(x_noisy, t_hat, ref_pos, ref_charge, ref_mask, ref_element,
              ref_atom_name_chars, ref_space_uid, tok_idx, s_inputs, z_trunk, relpos)

    assert out.shape == ref_out.shape, (out.shape, ref_out.shape)
    p = pcc(out, ref_out)
    assert p > 0.98, f"PCC {p:.5f} too low (n_tokens={n_tokens}, N={N})"


@pytest.mark.parametrize("n_atoms", [80, 200])
def test_swa_atom_transformer(n_atoms):
    ref = make_swa_atom_transformer(n_blocks=3)
    q = torch.randn(1, n_atoms, 128)
    c = torch.randn(1, n_atoms, 128)
    ref_pos = torch.randn(1, n_atoms, 3)
    uid = torch.randint(0, 8, (1, n_atoms))
    cos, sin = ref._build_3d_rope(ref_pos, uid)
    ref_out = ref(q, c, (cos, sin))

    mod = tt_ef2.SWAAtomTransformer(n_blocks=3, n_heads=4, swa_window_size=128, d_atom=128)
    mod.load_state_dict(ref.state_dict(), strict=False)
    out = mod(q, c, ref_pos, uid)

    assert out.shape == ref_out.shape
    p = pcc(out, ref_out)
    assert p > 0.98, f"PCC {p:.5f} too low (n_atoms={n_atoms})"


@pytest.mark.parametrize("seq_len", [37, 64])
def test_diffusion_conditioning(seq_len):
    ref = make_diffusion_conditioning()
    s_inputs = torch.randn(1, seq_len, 451)
    z_trunk = torch.randn(1, seq_len, seq_len, 256)
    relpos = torch.randn(1, seq_len, seq_len, 256)
    t_hat = torch.tensor([12.0])
    s_ref, z_ref = ref(t_hat, s_inputs, None, z_trunk, relpos)

    mod = tt_ef2.DiffusionConditioning(sigma_data=16.0)
    mod.load_state_dict(ref.state_dict(), strict=False)
    s_out, z_out = mod(t_hat, s_inputs, z_trunk, relpos)

    assert s_out.shape == s_ref.shape and z_out.shape == z_ref.shape
    ps, pz = pcc(s_out, s_ref), pcc(z_out, z_ref)
    assert ps > 0.99 and pz > 0.99, f"s PCC {ps:.5f}, z PCC {pz:.5f}"


@pytest.mark.parametrize("num_blocks,seq_len", [(1, 32), (4, 64), (12, 48)])
def test_diffusion_token_transformer(num_blocks, seq_len):
    d_model = DIFFUSION_TOKEN["d_model"]
    d_pair = DIFFUSION_TOKEN["d_pair"]
    n_heads = DIFFUSION_TOKEN["num_heads"]

    ref = make_diffusion_transformer(num_blocks=num_blocks)
    a = torch.randn(1, seq_len, d_model)
    s = torch.randn(1, seq_len, d_model)
    z = torch.randn(1, seq_len, seq_len, d_pair)
    ref_out, _ = ref(a, s, z, beta=0.0)

    mod = tt_ef2.DiffusionTransformer(num_heads=n_heads, num_blocks=num_blocks)
    mod.load_state_dict(ref.state_dict(), strict=False)
    out = mod(a, s, z)

    assert out.shape == ref_out.shape, (out.shape, ref_out.shape)
    p = pcc(out, ref_out)
    assert p > 0.98, f"PCC {p:.5f} too low (blocks={num_blocks}, L={seq_len})"
