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


# --- TriangleAttention (second parity gate) -----------------------------------
def make_triangle_attention(c_in=128, c_hidden=32, no_heads=4, starting=True, seed=0):
    """OpenFold TriangleAttention (starting/ending node), random weights."""
    from protenix.openfold_local.model.triangular_attention import TriangleAttention

    torch.manual_seed(seed)
    mod = TriangleAttention(c_in=c_in, c_hidden=c_hidden, no_heads=no_heads,
                            starting=starting).eval()
    for p in mod.parameters():
        p.data.normal_(0.0, 0.5)
    return mod, mod.state_dict()


def remap_triangle_attention(ref_sd: dict) -> dict:
    """OpenFold TriangleAttention -> tt-bio TriangleAttention: strip the `mha.`
    prefix from the q/k/v/o/g projections; layer_norm + linear (pair bias) map
    directly. tt-bio fuses q/k/v and folds the 1/sqrt(d) scale into `linear`."""
    out = {}
    for k, v in ref_sd.items():
        out[k[len("mha."):] if k.startswith("mha.") else k] = v
    return out


def run_reference_triangle_attention(mod, x, mask=None):
    """Forward OpenFold TriangleAttention. x: [B,L,L,c_in]."""
    with torch.no_grad():
        if mask is None:
            mask = torch.ones(x.shape[:-1], dtype=x.dtype)
        return mod(x, mask=mask)


# --- Transition (SwiGLU; used as pair_transition + single_transition) ---------
def make_transition(c_in=128, n=4, seed=0):
    from protenix.model.modules.primitives import Transition

    torch.manual_seed(seed)
    mod = Transition(c_in=c_in, n=n).eval()
    for p in mod.parameters():
        p.data.normal_(0.0, 0.5)
    return mod, mod.state_dict()


def remap_transition(ref_sd: dict) -> dict:
    """Protenix Transition -> tt-bio Transition. Both: out = linear(silu(a)*b),
    a/b = linear_{a,b}(LN(x)). tt-bio folds silu into fc1."""
    return {
        "norm.weight": ref_sd["layernorm1.weight"],
        "norm.bias": ref_sd["layernorm1.bias"],
        "fc1.weight": ref_sd["linear_no_bias_a.weight"],  # gets silu in tt-bio
        "fc2.weight": ref_sd["linear_no_bias_b.weight"],
        "fc3.weight": ref_sd["linear_no_bias.weight"],
    }


def run_reference_transition(mod, x):
    with torch.no_grad():
        return mod(x)


# --- AttentionPairBias (Pairformer single-attn-with-pair-bias, has_s=False) ---
def make_attention_pair_bias(c_a=384, c_z=128, n_heads=16, seed=0):
    from protenix.model.modules.transformer import AttentionPairBias as RefAPB

    torch.manual_seed(seed)
    mod = RefAPB(has_s=False, create_offset_ln_z=True, n_heads=n_heads,
                 c_a=c_a, c_z=c_z).eval()
    for p in mod.parameters():
        p.data.normal_(0.0, 0.3)
    return mod, mod.state_dict()


def remap_attention_pair_bias(ref_sd: dict) -> dict:
    """Protenix AttentionPairBias -> tt-bio AttentionPairBias (atom_level=False,
    compute_pair_bias=True). The input-`a` LayerNorm is applied externally by the
    caller (tt-bio does it via PairformerLayer.pre_norm_s); here the test feeds
    layernorm_a(a). Verified raw mapping (PCC 0.99986) — q has a bias."""
    return {
        "proj_q.weight": ref_sd["attention.linear_q.weight"],
        "proj_q.bias": ref_sd["attention.linear_q.bias"],
        "proj_k.weight": ref_sd["attention.linear_k.weight"],
        "proj_v.weight": ref_sd["attention.linear_v.weight"],
        "proj_g.weight": ref_sd["attention.linear_g.weight"],
        "proj_o.weight": ref_sd["attention.linear_o.weight"],
        "proj_z.0.weight": ref_sd["layernorm_z.weight"],
        "proj_z.0.bias": ref_sd["layernorm_z.bias"],
        "proj_z.1.weight": ref_sd["linear_nobias_z.weight"],
    }


# --- Full PairformerBlock (composition of the 4 verified sub-modules) ---------
def make_pairformer_block(c_z=128, c_s=384, n_heads=16, c_hidden_mul=128,
                          c_hidden_pair_att=32, no_heads_pair=4, seed=0):
    from protenix.model.modules.pairformer import PairformerBlock

    torch.manual_seed(seed)
    mod = PairformerBlock(n_heads=n_heads, c_z=c_z, c_s=c_s,
                          c_hidden_mul=c_hidden_mul,
                          c_hidden_pair_att=c_hidden_pair_att,
                          no_heads_pair=no_heads_pair).eval()
    for p in mod.parameters():
        p.data.normal_(0.0, 0.3)
    return mod, mod.state_dict()


def remap_pairformer_block(sd: dict) -> dict:
    """Protenix PairformerBlock -> tt-bio PairformerLayer flat state_dict.
    tt-bio scopes: tri_mul_out/in (fused remap), tri_att_start/end (DIRECT — the
    `mha.` prefix is stripped by scope), transition_z<-pair_transition,
    attention<-attention_pair_bias.attention, pre_norm_s<-attention_pair_bias.
    layernorm_a, transition_s<-single_transition."""
    def sub(p):
        return {k[len(p) + 1:]: v for k, v in sd.items() if k.startswith(p + ".")}

    out = {}
    for k, v in remap_triangle_multiplication(sub("tri_mul_out")).items():
        out[f"tri_mul_out.{k}"] = v
    for k, v in remap_triangle_multiplication(sub("tri_mul_in")).items():
        out[f"tri_mul_in.{k}"] = v
    for k, v in sub("tri_att_start").items():   # direct (scope strips mha.)
        out[f"tri_att_start.{k}"] = v
    for k, v in sub("tri_att_end").items():
        out[f"tri_att_end.{k}"] = v
    for k, v in remap_transition(sub("pair_transition")).items():
        out[f"transition_z.{k}"] = v
    apb = sub("attention_pair_bias")
    out["pre_norm_s.weight"] = apb["layernorm_a.weight"]
    out["pre_norm_s.bias"] = apb["layernorm_a.bias"]
    for k, v in remap_attention_pair_bias(apb).items():
        out[f"attention.{k}"] = v
    for k, v in remap_transition(sub("single_transition")).items():
        out[f"transition_s.{k}"] = v
    return out


def run_reference_pairformer_block(mod, s, z):
    with torch.no_grad():
        pair_mask = torch.ones(z.shape[:-1], dtype=z.dtype)  # [B,L,L]; ones == no mask
        return mod(s, z, pair_mask)  # -> (s, z)


# --- OuterProductMean (MSA -> pair) -------------------------------------------
def make_outer_product_mean(c_m=128, c_z=128, c_hidden=32, seed=0):
    from protenix.openfold_local.model.outer_product_mean import OuterProductMean as RefOPM

    torch.manual_seed(seed)
    mod = RefOPM(c_m=c_m, c_z=c_z, c_hidden=c_hidden).eval()
    for p in mod.parameters():
        p.data.normal_(0.0, 0.3)
    return mod, mod.state_dict()


def remap_outer_product_mean(ref_sd: dict) -> dict:
    """OpenFold OuterProductMean -> tt-bio OuterProductMean. Direct (verified
    PCC 0.99962; the c_hidden^2 flatten order matches, no permute needed)."""
    return {
        "norm.weight": ref_sd["layer_norm.weight"],
        "norm.bias": ref_sd["layer_norm.bias"],
        "proj_a.weight": ref_sd["linear_1.weight"],
        "proj_b.weight": ref_sd["linear_2.weight"],
        "proj_o.weight": ref_sd["linear_out.weight"],
        "proj_o.bias": ref_sd["linear_out.bias"],
    }


def run_reference_outer_product_mean(mod, m):
    with torch.no_grad():
        return mod(m, mask=torch.ones(m.shape[:-1], dtype=m.dtype))  # [B,L,L,c_z]


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
