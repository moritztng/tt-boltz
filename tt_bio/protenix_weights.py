"""Protenix-v2 -> tt-bio weight-name remaps (pure dict/tensor functions).

Protenix-v2 is the same AlphaFold3 family as Boltz-2, so its trunk/MSA/template
modules map onto the existing tt_bio.tenstorrent primitives (TriangleMultiplication,
TriangleAttention, Transition, AttentionPairBias, OuterProductMean,
PairWeightedAveraging, PairformerLayer). These remaps translate Protenix/OpenFold
checkpoint key names onto each primitive's expected state_dict. Every mapping was
validated on-device vs the real v2 reference (PCC > 0.98; see docs/porting-protenix-v2.md
and tests/test_protenix*.py). No protenix import -- pure torch.cat/rename on tensors.
"""

from __future__ import annotations

import torch


def remap_triangle_multiplication(ref_sd: dict) -> dict:
    """OpenFold TriangleMultiplication{Outgoing,Incoming} -> tt-bio fused layout
    (g_in/p_in rows = [a-side ; b-side])."""
    return {
        "norm_in.weight": ref_sd["layer_norm_in.weight"],
        "norm_in.bias": ref_sd["layer_norm_in.bias"],
        "norm_out.weight": ref_sd["layer_norm_out.weight"],
        "norm_out.bias": ref_sd["layer_norm_out.bias"],
        "g_in.weight": torch.cat([ref_sd["linear_a_g.weight"], ref_sd["linear_b_g.weight"]], dim=0),
        "p_in.weight": torch.cat([ref_sd["linear_a_p.weight"], ref_sd["linear_b_p.weight"]], dim=0),
        "g_out.weight": ref_sd["linear_g.weight"],
        "p_out.weight": ref_sd["linear_z.weight"],
    }


def remap_adaln(ref_sd: dict) -> dict:
    """Protenix AdaptiveLayerNorm -> tt-bio AdaLN. a = sigmoid(linear_s(LN_s(s)))*LN_a(a)
    + linear_nobias_s(LN_s(s)); LN_a has no affine. Validated on-device (PCC 0.999996)."""
    return {
        "s_norm.weight": ref_sd["layernorm_s.weight"],
        "s_scale.weight": ref_sd["linear_s.weight"],
        "s_scale.bias": ref_sd["linear_s.bias"],
        "s_bias.weight": ref_sd["linear_nobias_s.weight"],
    }


def remap_transition(ref_sd: dict) -> dict:
    """Protenix Transition -> tt-bio Transition (silu folded into fc1)."""
    return {
        "norm.weight": ref_sd["layernorm1.weight"],
        "norm.bias": ref_sd["layernorm1.bias"],
        "fc1.weight": ref_sd["linear_no_bias_a.weight"],
        "fc2.weight": ref_sd["linear_no_bias_b.weight"],
        "fc3.weight": ref_sd["linear_no_bias.weight"],
    }


def remap_attention_pair_bias(ref_sd: dict) -> dict:
    """Protenix AttentionPairBias -> tt-bio AttentionPairBias (atom_level=False,
    compute_pair_bias=True). The input-`a` LayerNorm is applied externally."""
    return {
        "proj_q.weight": ref_sd["attention.linear_q.weight"],
        "proj_q.bias": ref_sd["attention.linear_q.bias"],
        "proj_k.weight": ref_sd["attention.linear_k.weight"],
        "proj_v.weight": ref_sd["attention.linear_v.weight"],
        "proj_g.weight": ref_sd["attention.linear_g.weight"],
        "proj_o.weight": ref_sd["attention.linear_o.weight"],
        "proj_z.0.weight": ref_sd["layernorm_z.weight"],
        "proj_z.0.bias": ref_sd.get("layernorm_z.bias", torch.zeros_like(ref_sd["layernorm_z.weight"])),
        "proj_z.1.weight": ref_sd["linear_nobias_z.weight"],
    }


def remap_pairformer_block(sd: dict) -> dict:
    """Protenix PairformerBlock -> tt-bio PairformerLayer flat state_dict."""
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


def remap_outer_product_mean(ref_sd: dict) -> dict:
    """OpenFold OuterProductMean -> tt-bio OuterProductMean (direct)."""
    return {
        "norm.weight": ref_sd["layer_norm.weight"],
        "norm.bias": ref_sd["layer_norm.bias"],
        "proj_a.weight": ref_sd["linear_1.weight"],
        "proj_b.weight": ref_sd["linear_2.weight"],
        "proj_o.weight": ref_sd["linear_out.weight"],
        "proj_o.bias": ref_sd["linear_out.bias"],
    }


def remap_pair_weighted_averaging(ref_sd: dict) -> dict:
    """Protenix MSAPairWeightedAveraging -> tt-bio PairWeightedAveraging."""
    return {
        "norm_m.weight": ref_sd["layernorm_m.weight"],
        "norm_m.bias": ref_sd["layernorm_m.bias"],
        "norm_z.weight": ref_sd["layernorm_z.weight"],
        "norm_z.bias": ref_sd["layernorm_z.bias"],
        "proj_m.weight": ref_sd["linear_no_bias_mv.weight"],
        "proj_g.weight": ref_sd["linear_no_bias_mg.weight"],
        "proj_z.weight": ref_sd["linear_no_bias_z.weight"],
        "proj_o.weight": ref_sd["linear_no_bias_out.weight"],
    }


def remap_msa_pair_stack(ps_sd: dict) -> dict:
    """Pair-only remap for the MSA block / template pair_stack (PairformerBlock c_s=0)."""
    def s(p):
        return {k[len(p) + 1:]: v for k, v in ps_sd.items() if k.startswith(p + ".")}
    out = {}
    for k, v in remap_triangle_multiplication(s("tri_mul_out")).items():
        out[f"tri_mul_out.{k}"] = v
    for k, v in remap_triangle_multiplication(s("tri_mul_in")).items():
        out[f"tri_mul_in.{k}"] = v
    for k, v in s("tri_att_start").items():
        out[f"tri_att_start.{k}"] = v
    for k, v in s("tri_att_end").items():
        out[f"tri_att_end.{k}"] = v
    for k, v in remap_transition(s("pair_transition")).items():
        out[f"transition_z.{k}"] = v
    return out
