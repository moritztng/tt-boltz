"""Protenix-v2 reference harness for the ttnn port (branch protenix-v2).

Same idiom as tests/esmfold2_reference.py: import the *reference* PyTorch
modules from the pip-installed `protenix` package (installed --no-deps so it
never clobbers the ttnn torch build), build them with random weights, and
provide weight-name remaps from Protenix/OpenFold naming onto the existing
tt-bio modules (built for Boltz-2 — the same AlphaFold3 family). Each tt-bio
module is then validated against its Protenix reference at PCC > 0.98.

Phase 1 of docs/porting-protenix-v2.md. This file holds the reference builders
and remaps; tests/test_protenix.py runs the on-device parity checks.
"""

from __future__ import annotations

import torch

# --- Confirmed Protenix-v2 architecture (from protenix.model + config) --------
# AF3-standard, identical block graph to Boltz-2 (already ported in tt-bio).
PROTENIX_ARCH = {
    "c_s": 384,          # single representation
    "c_z": 128,          # pair representation
    "c_hidden_mul": 128,  # TriangleMultiplication hidden
    "c_hidden_pair_att": 32,  # TriangleAttention hidden
    "n_heads": 16,       # AttentionPairBias heads
    "no_heads_pair": 4,  # TriangleAttention heads
    "pairformer_blocks": 48,
    "msa_blocks": 4,
    "diffusion_token_blocks": 24,
    "diffusion_atom_blocks": 3,   # atom encoder / decoder
    # Note: Protenix-v2 ("enhanced capacity", ~464M) raises some of these; the
    # exact v2 dims are read from the v2 checkpoint config in Phase 2.
}


# --- TriangleMultiplication (first parity gate) -------------------------------
def make_triangle_multiplication(c_z=128, c_hidden=128, outgoing=True, seed=0):
    """OpenFold-style TriangleMultiplication{Outgoing,Incoming} (what Protenix's
    PairformerBlock uses), random weights. Returns (module, state_dict)."""
    from protenix.openfold_local.model.triangular_multiplicative_update import (
        TriangleMultiplicationIncoming,
        TriangleMultiplicationOutgoing,
    )

    torch.manual_seed(seed)
    cls = TriangleMultiplicationOutgoing if outgoing else TriangleMultiplicationIncoming
    mod = cls(c_z=c_z, c_hidden=c_hidden).eval()
    # OpenFold zero-inits the final linear (init="final") and gates, so a fresh
    # module is identically zero. For a parity test we need a non-trivial
    # function — randomize every parameter (both sides use these same weights).
    for p in mod.parameters():
        p.data.normal_(0.0, 0.5)
    return mod, mod.state_dict()


def remap_triangle_multiplication(ref_sd: dict) -> dict:
    """Map OpenFold TriangleMultiplication weights -> tt-bio's fused
    TriangleMultiplication layout (tt_bio.tenstorrent.TriangleMultiplication).

    tt-bio packs the per-side gate/value projections into fused g_in / p_in
    (rows = [a-side (c_hidden), b-side (c_hidden)]) and computes
    a = linear_a_p(z) * sigmoid(linear_a_g(z)) — identical to OpenFold's
    a = a * sigmoid(linear_a_g(z)) * linear_a_p(z).
    """
    return {
        "norm_in.weight": ref_sd["layer_norm_in.weight"],
        "norm_in.bias": ref_sd["layer_norm_in.bias"],
        "norm_out.weight": ref_sd["layer_norm_out.weight"],
        "norm_out.bias": ref_sd["layer_norm_out.bias"],
        # fused: rows [a-gate ; b-gate] and [a-proj ; b-proj]
        "g_in.weight": torch.cat([ref_sd["linear_a_g.weight"], ref_sd["linear_b_g.weight"]], dim=0),
        "p_in.weight": torch.cat([ref_sd["linear_a_p.weight"], ref_sd["linear_b_p.weight"]], dim=0),
        "g_out.weight": ref_sd["linear_g.weight"],   # output gate
        "p_out.weight": ref_sd["linear_z.weight"],   # output projection
    }


def run_reference_triangle_multiplication(mod, z, mask=None):
    """Forward the OpenFold reference. z: [B,L,L,c_z]; mask: [B,L,L] or None."""
    with torch.no_grad():
        if mask is None:
            mask = torch.ones(z.shape[:-1], dtype=z.dtype)
        return mod(z, mask=mask)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


if __name__ == "__main__":
    # CPU self-check: reference builds + runs, remap produces correct shapes.
    c_z, c_h, L = 128, 128, 16
    mod, sd = make_triangle_multiplication(c_z, c_h, outgoing=True)
    z = torch.randn(1, L, L, c_z)
    out = run_reference_triangle_multiplication(mod, z)
    print(f"reference out: shape={tuple(out.shape)} finite={torch.isfinite(out).all().item()}")
    rm = remap_triangle_multiplication(sd)
    print("remap shapes:", {k: tuple(v.shape) for k, v in rm.items()})
    assert rm["g_in.weight"].shape == (2 * c_h, c_z), rm["g_in.weight"].shape
    assert rm["p_in.weight"].shape == (2 * c_h, c_z), rm["p_in.weight"].shape
    assert rm["g_out.weight"].shape == (c_z, c_z)
    assert rm["p_out.weight"].shape == (c_z, c_h)
    print("OK: reference runs + remap shapes match tt-bio TriangleMultiplication")
