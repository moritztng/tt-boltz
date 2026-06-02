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
    make_diffusion_transformer,
    make_folding_trunk,
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
