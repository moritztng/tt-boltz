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

# weight remaps live in the package (single source); re-exported here for the
# reference-parity tests. remap_adaptive_layernorm is the AdaLN remap.
from tt_bio.protenix_weights import (  # noqa: E402
    remap_triangle_multiplication, remap_transition, remap_attention_pair_bias,
    remap_pairformer_block, remap_outer_product_mean, remap_pair_weighted_averaging,
    remap_msa_pair_stack, remap_adaln as remap_adaptive_layernorm,
)

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


def run_reference_outer_product_mean(mod, m):
    with torch.no_grad():
        return mod(m, mask=torch.ones(m.shape[:-1], dtype=m.dtype))  # [B,L,L,c_z]


# --- MSAPairWeightedAveraging (pair-biased MSA averaging) ---------------------
def make_pair_weighted_averaging(c_m=64, c=32, c_z=128, n_heads=8, seed=0):
    from protenix.model.modules.pairformer import MSAPairWeightedAveraging

    torch.manual_seed(seed)
    mod = MSAPairWeightedAveraging(c_m=c_m, c=c, c_z=c_z, n_heads=n_heads).eval()
    for p in mod.parameters():
        p.data.normal_(0.0, 0.3)
    return mod, mod.state_dict()


def run_reference_pair_weighted_averaging(mod, m, z):
    with torch.no_grad():
        return mod(m, z)


# --- Full MSABlock (Protenix-ordered assembly of verified sub-modules) --------
# NOTE: Protenix reference forwards mutate m/z IN PLACE (inplace_safe), so any
# parity test must clone the inputs before calling the reference and feed the
# pristine clones to the tt-bio side.
def make_msa_block(c_m=64, c_z=128, c_hidden=32, seed=0):
    from protenix.model.modules.pairformer import MSABlock

    torch.manual_seed(seed)
    mod = MSABlock(c_m=c_m, c_z=c_z, c_hidden=c_hidden, is_last_block=False).eval()
    for p in mod.parameters():
        p.data.normal_(0.0, 0.3)
    return mod, mod.state_dict()


def run_reference_msa_block(mod, m, z):
    """Reference MSABlock forward. MUTATES m,z in place — pass clones."""
    with torch.no_grad():
        return mod(m, z, torch.ones(z.shape[:-1], dtype=z.dtype))  # (m, z)


# --- AdaptiveLayerNorm (diffusion conditioning norm) --------------------------
def make_adaptive_layernorm(c_a=768, c_s=384, seed=0):
    from protenix.model.modules.primitives import AdaptiveLayerNorm

    torch.manual_seed(seed)
    mod = AdaptiveLayerNorm(c_a=c_a, c_s=c_s).eval()
    for p in mod.parameters():
        p.data.normal_(0.0, 0.3)
    return mod, mod.state_dict()


def run_reference_adaptive_layernorm(mod, a, s):
    with torch.no_grad():
        return mod(a, s)


# --- ConditionedTransitionBlock (Protenix-specific assembly) ------------------
# tt-bio's ConditionedTransitionBlock (Boltz-2) computes b = silu(swish)*gates*
# a_to_b (3 factors); Protenix's is b = silu(a1)*a2 (2 factors) — NOT a drop-in.
# So assemble Protenix's from the verified AdaLN + raw ttnn linears (see test).
def make_conditioned_transition_block(c_a=768, c_s=384, n=2, seed=0):
    from protenix.model.modules.transformer import ConditionedTransitionBlock

    torch.manual_seed(seed)
    mod = ConditionedTransitionBlock(c_a=c_a, c_s=c_s, n=n).eval()
    for p in mod.parameters():
        p.data.normal_(0.0, 0.3)
    return mod, mod.state_dict()


def run_reference_conditioned_transition_block(mod, a, s):
    with torch.no_grad():
        return mod(a, s)


# --- DiffusionTransformerBlock (token DiT: AdaLN-attn + CTB) -------------------
def make_diffusion_transformer_block(c_a=768, c_s=384, c_z=128, n_heads=16, seed=0):
    from protenix.model.modules.transformer import DiffusionTransformerBlock

    torch.manual_seed(seed)
    mod = DiffusionTransformerBlock(c_a=c_a, c_s=c_s, c_z=c_z, n_heads=n_heads).eval()
    for p in mod.parameters():
        p.data.normal_(0.0, 0.3)
    return mod, mod.state_dict()


def run_reference_diffusion_transformer_block(mod, a, s, z):
    with torch.no_grad():
        return mod(a, s, z)[0]  # out_a


# --- DistogramHead -----------------------------------------------------------
def make_distogram_head(c_z=128, no_bins=64, seed=0):
    from protenix.model.modules.head import DistogramHead

    torch.manual_seed(seed)
    mod = DistogramHead(c_z=c_z, no_bins=no_bins).eval()
    for p in mod.parameters():
        p.data.normal_(0.0, 0.3)
    return mod, mod.state_dict()


def remap_distogram_head(ref_sd: dict) -> dict:
    """Protenix: logits=linear(z); out=logits+logits.T -> W(z+zT)+2b.
    tt-bio esmfold2 DistogramHead: linear(z+zT) -> W(z+zT)+b. So bias must double."""
    return {"weight": ref_sd["linear.weight"], "bias": 2.0 * ref_sd["linear.bias"]}


def run_reference_distogram_head(mod, z):
    with torch.no_grad():
        return mod(z)


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
