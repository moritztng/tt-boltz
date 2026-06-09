"""On-device parity tests for the Protenix-v2 ttnn port (branch protenix-v2).

Each test builds a Protenix/OpenFold reference module with random weights,
remaps its weights onto the existing tt-bio module (built for Boltz-2 — same
AF3 family), runs the tt-bio module on the TT device, and asserts PCC > 0.98.
Idiom mirrors tests/test_esmfold2.py. See docs/porting-protenix-v2.md.
"""

import os
import sys

import pytest
import torch
import ttnn

sys.path.insert(0, os.path.dirname(__file__))
from protenix_reference import (  # noqa: E402
    make_triangle_attention,
    make_triangle_multiplication,
    pcc,
    remap_triangle_attention,
    remap_triangle_multiplication,
    run_reference_triangle_attention,
    run_reference_triangle_multiplication,
)

from tt_bio.tenstorrent import (  # noqa: E402
    TriangleAttention,
    TriangleMultiplication,
    get_device,
)

torch.set_grad_enabled(False)


def _ck(dev):
    cls = (ttnn.types.WormholeComputeKernelConfig
           if dev.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    return cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
               fp32_dest_acc_en=True, packer_l1_acc=True)


# Protenix uses OpenFold TriangleMultiplication{Outgoing,Incoming}; tt-bio's
# `ending` flag is False=outgoing, True=incoming.
@pytest.mark.parametrize("outgoing,ending", [(True, False), (False, True)])
def test_triangle_multiplication_parity(outgoing, ending):
    c_z, c_hidden, L = 128, 128, 64
    mod, sd = make_triangle_multiplication(c_z, c_hidden, outgoing=outgoing, seed=0)
    z = torch.randn(1, L, L, c_z)
    ref = run_reference_triangle_multiplication(mod, z).float()

    dev = get_device()
    tm = TriangleMultiplication(
        ending=ending,
        state_dict=remap_triangle_multiplication(sd),
        compute_kernel_config=_ck(dev),
    )
    x = ttnn.from_torch(z, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    out = torch.Tensor(ttnn.to_torch(tm(x))).float()
    p = pcc(out, ref)
    assert p > 0.98, f"PCC {p:.5f} (outgoing={outgoing}, ending={ending})"


# Protenix TriangleAttention starting/ending node -> tt-bio `ending` = not starting.
@pytest.mark.parametrize("starting,ending", [(True, False), (False, True)])
def test_triangle_attention_parity(starting, ending):
    c_in, c_hidden, no_heads, L = 128, 32, 4, 64
    mod, sd = make_triangle_attention(c_in, c_hidden, no_heads, starting=starting, seed=0)
    x = torch.randn(1, L, L, c_in)
    ref = run_reference_triangle_attention(mod, x).float()

    dev = get_device()
    ta = TriangleAttention(
        head_dim=c_hidden, n_heads=no_heads, ending=ending,
        state_dict=remap_triangle_attention(sd), compute_kernel_config=_ck(dev),
    )
    xt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    out = torch.Tensor(ttnn.to_torch(ta(xt))).float()
    p = pcc(out, ref)
    assert p > 0.98, f"PCC {p:.5f} (starting={starting}, ending={ending})"
