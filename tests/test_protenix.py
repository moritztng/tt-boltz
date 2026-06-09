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
    make_attention_pair_bias,
    make_outer_product_mean,
    make_adaptive_layernorm,
    make_conditioned_transition_block,
    make_diffusion_transformer_block,
    make_distogram_head,
    make_msa_block,
    make_pair_weighted_averaging,
    make_pairformer_block,
    make_transition,
    make_triangle_attention,
    make_triangle_multiplication,
    pcc,
    remap_attention_pair_bias,
    remap_outer_product_mean,
    remap_adaptive_layernorm,
    remap_msa_pair_stack,
    remap_pair_weighted_averaging,
    remap_pairformer_block,
    remap_transition,
    remap_triangle_attention,
    remap_triangle_multiplication,
    run_reference_outer_product_mean,
    run_reference_adaptive_layernorm,
    run_reference_conditioned_transition_block,
    run_reference_diffusion_transformer_block,
    run_reference_distogram_head,
    remap_distogram_head,
    run_reference_msa_block,
    run_reference_pair_weighted_averaging,
    run_reference_pairformer_block,
    run_reference_transition,
    run_reference_triangle_attention,
    run_reference_triangle_multiplication,
)

from tt_bio.tenstorrent import (  # noqa: E402
    CORE_GRID_MAIN,
    AttentionPairBias,
    AdaLN,
    OuterProductMean,
    PairWeightedAveraging,
    PairformerLayer,
    Transition,
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


# Protenix Transition (SwiGLU) -> tt-bio Transition.
@pytest.mark.parametrize("c_in", [128, 384])
def test_transition_parity(c_in):
    L, n = 64, 4
    mod, sd = make_transition(c_in=c_in, n=n, seed=0)
    x = torch.randn(1, L, L, c_in) if c_in == 128 else torch.randn(1, L, c_in)
    ref = run_reference_transition(mod, x).float()

    dev = get_device()
    tr = Transition(state_dict=remap_transition(sd), compute_kernel_config=_ck(dev))
    xt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    out = torch.Tensor(ttnn.to_torch(tr(xt))).float()
    p = pcc(out, ref)
    assert p > 0.98, f"PCC {p:.5f} (c_in={c_in})"


# Protenix AttentionPairBias (Pairformer, has_s=False) -> tt-bio AttentionPairBias.
# Input-a LayerNorm is applied externally (tt-bio does it via PairformerLayer.pre_norm_s).
def test_attention_pair_bias_parity():
    c_a, c_z, n_heads, L = 384, 128, 16, 64
    head_dim = c_a // n_heads
    mod, sd = make_attention_pair_bias(c_a, c_z, n_heads, seed=0)
    a = torch.randn(1, L, c_a)
    z = torch.randn(1, L, L, c_z)
    ref = mod(a, None, z).float()
    a_normed = mod.layernorm_a(a)  # external a-norm

    dev = get_device()
    apb = AttentionPairBias(head_dim, n_heads, True, False,
                            remap_attention_pair_bias(sd), _ck(dev))
    s_t = ttnn.from_torch(a_normed, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    z_t = ttnn.from_torch(z, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    out = torch.Tensor(ttnn.to_torch(apb(s_t, z_t))).float()
    p = pcc(out, ref)
    assert p > 0.98, f"PCC {p:.5f}"


# Full PairformerBlock: compose the 4 verified sub-modules + pre-norms/residuals.
def test_pairformer_block_parity():
    c_z, c_s, L = 128, 384, 64
    mod, sd = make_pairformer_block(c_z=c_z, c_s=c_s, seed=0)
    s = torch.randn(1, L, c_s)
    z = torch.randn(1, L, L, c_z)
    s_ref, z_ref = run_reference_pairformer_block(mod, s, z)
    s_ref, z_ref = s_ref.float(), z_ref.float()

    dev = get_device()
    layer = PairformerLayer(
        tri_att_head_dim=32, tri_att_n_heads=4, att_head_dim=24, att_n_heads=16,
        transform_s=True, state_dict=remap_pairformer_block(sd),
        compute_kernel_config=_ck(dev),
    )
    s_t = ttnn.from_torch(s, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    z_t = ttnn.from_torch(z, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    s_out, z_out = layer(s_t, z_t)
    s_o = torch.Tensor(ttnn.to_torch(s_out)).float()
    z_o = torch.Tensor(ttnn.to_torch(z_out)).float()
    ps, pz = pcc(s_o, s_ref), pcc(z_o, z_ref)
    assert ps > 0.98 and pz > 0.98, f"PCC s={ps:.5f} z={pz:.5f}"


# OuterProductMean (MSA -> pair).
def test_outer_product_mean_parity():
    c_m, c_z, c_hidden, S, L = 128, 128, 32, 8, 32
    mod, sd = make_outer_product_mean(c_m, c_z, c_hidden, seed=0)
    m = torch.randn(1, S, L, c_m)
    ref = run_reference_outer_product_mean(mod, m).float()[0]
    dev = get_device()
    opm = OuterProductMean(remap_outer_product_mean(sd), _ck(dev))
    mt = ttnn.from_torch(m, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    out = torch.Tensor(ttnn.to_torch(opm(mt, None, None))).float()
    if out.dim() == 4:
        out = out[0]
    p = pcc(out, ref)
    assert p > 0.98, f"PCC {p:.5f}"


# MSAPairWeightedAveraging.
def test_pair_weighted_averaging_parity():
    c_m, c, c_z, n_heads, S, L = 64, 32, 128, 8, 8, 32
    mod, sd = make_pair_weighted_averaging(c_m, c, c_z, n_heads, seed=0)
    m = torch.randn(1, S, L, c_m)
    z = torch.randn(1, L, L, c_z)
    ref = run_reference_pair_weighted_averaging(mod, m, z).float()
    dev = get_device()
    pwa = PairWeightedAveraging(head_dim=c, n_heads=n_heads,
                                state_dict=remap_pair_weighted_averaging(sd),
                                compute_kernel_config=_ck(dev))
    m_t = ttnn.from_torch(m, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    z_t = ttnn.from_torch(z, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    out = torch.Tensor(ttnn.to_torch(pwa(m_t, z_t))).float()
    while out.dim() > ref.dim():
        out = out[0]
    if out.shape != ref.shape and out.dim() == ref.dim():
        out = out.reshape(ref.shape)
    p = pcc(out, ref)
    assert p > 0.98, f"PCC {p:.5f} (shapes out={tuple(out.shape)} ref={tuple(ref.shape)})"


# Full MSABlock: Protenix-ordered assembly of verified sub-modules (OPM-first).
# Reference mutates m,z in place -> clone before the reference call.
def test_msa_block_parity():
    c_m, c_z, S, L = 64, 128, 8, 32
    mod, sd = make_msa_block(c_m, c_z, 32, seed=0)
    m = torch.randn(1, S, L, c_m); z = torch.randn(1, L, L, c_z)
    m0, z0 = m.clone(), z.clone()  # reference mutates m,z in place
    ref_m, ref_z = run_reference_msa_block(mod, m, z)
    ref_m, ref_z = ref_m.float(), ref_z.float()

    def sub(d, p):
        return {k[len(p) + 1:]: v for k, v in d.items() if k.startswith(p + ".")}
    dev = get_device(); ck = _ck(dev)
    opm = OuterProductMean(remap_outer_product_mean(sub(sd, "outer_product_mean_msa")), ck)
    pwa = PairWeightedAveraging(8, 8, remap_pair_weighted_averaging(sub(sd, "msa_stack.msa_pair_weighted_averaging")), ck)
    tm = Transition(remap_transition(sub(sd, "msa_stack.transition_m")), ck)
    pl = PairformerLayer(32, 4, None, None, False, remap_msa_pair_stack(sub(sd, "pair_stack")), ck)
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    m_t = ft(m0)
    z1 = ttnn.add(ft(z0), opm(m_t, None, None))                       # z += OPM(m)
    m_t = ttnn.add(m_t, ttnn.reshape(pwa(m_t, ttnn.clone(z1)), tuple(m_t.shape)))  # m += pwa(m,z)
    m_t = ttnn.add(m_t, ttnn.reshape(tm(m_t), tuple(m_t.shape)))      # m += transition(m)
    z_out = pl(None, z1)[1]                                            # z = pair_stack(z)
    mm = torch.Tensor(ttnn.to_torch(m_t)).float().reshape(ref_m.shape)
    zz = torch.Tensor(ttnn.to_torch(z_out)).float().reshape(ref_z.shape)
    pm_, pz_ = pcc(mm, ref_m), pcc(zz, ref_z)
    assert pm_ > 0.98 and pz_ > 0.98, f"m={pm_:.5f} z={pz_:.5f}"


# AdaptiveLayerNorm (diffusion conditioning).
def test_adaln_parity():
    c_a, c_s, L = 768, 384, 32
    mod, sd = make_adaptive_layernorm(c_a, c_s, seed=0)
    a = torch.randn(1, L, c_a); s = torch.randn(1, L, c_s)
    ref = run_reference_adaptive_layernorm(mod, a, s).float()
    dev = get_device()
    adaln = AdaLN(False, remap_adaptive_layernorm(sd), _ck(dev))
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    out = torch.Tensor(ttnn.to_torch(adaln(ft(a), ft(s)))).float().reshape(ref.shape)
    p = pcc(out, ref)
    assert p > 0.98, f"PCC {p:.5f}"


# Protenix ConditionedTransitionBlock: b = silu(a1)*a2 (2 factors) — assembled
# from the verified AdaLN + raw ttnn linears (tt-bio's CTB has an extra factor).
def test_conditioned_transition_block_parity():
    c_a, c_s, n, L = 768, 384, 2, 32
    mod, sd = make_conditioned_transition_block(c_a, c_s, n, seed=0)
    a = torch.randn(1, L, c_a); s = torch.randn(1, L, c_s)
    ref = run_reference_conditioned_transition_block(mod, a.clone(), s.clone()).float()
    dev = get_device(); ck = _ck(dev)
    def subd(p): return {k[len(p)+1:]: v for k, v in sd.items() if k.startswith(p + ".")}
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    ftt = lambda x: ttnn.from_torch(x.t(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    adaln = AdaLN(False, remap_adaptive_layernorm(subd("adaln")), ck)
    a1w, a2w, bw = ftt(sd["linear_nobias_a1.weight"]), ftt(sd["linear_nobias_a2.weight"]), ftt(sd["linear_nobias_b.weight"])
    lsw, lsb = ftt(sd["linear_s.weight"]), ft(sd["linear_s.bias"])
    an = adaln(ft(a), ft(s))
    a1 = ttnn.linear(an, a1w, compute_kernel_config=ck, core_grid=CORE_GRID_MAIN)
    a2 = ttnn.linear(an, a2w, compute_kernel_config=ck, core_grid=CORE_GRID_MAIN)
    b = ttnn.multiply(a2, a1, input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
    ba = ttnn.linear(b, bw, compute_kernel_config=ck, core_grid=CORE_GRID_MAIN)
    sg = ttnn.linear(ft(s), lsw, bias=lsb, compute_kernel_config=ck, core_grid=CORE_GRID_MAIN)
    out = ttnn.multiply(sg, ba, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
    ot = torch.Tensor(ttnn.to_torch(out)).float().reshape(ref.shape)
    p = pcc(ot, ref)
    assert p > 0.98, f"PCC {p:.5f}"


# Full diffusion token-transformer block: attn = sigmoid(linear_a_last(s)) *
# APB(AdaLN(a,s), z); out = attn + CTB(attn, s). Assembled from verified pieces.
def test_diffusion_transformer_block_parity():
    c_a, c_s, c_z, n_heads, L = 768, 384, 128, 16, 32
    hd = c_a // n_heads
    mod, sd = make_diffusion_transformer_block(c_a, c_s, c_z, n_heads, seed=0)
    a = torch.randn(1, L, c_a); s = torch.randn(1, L, c_s); z = torch.randn(1, L, L, c_z)
    out_ref = run_reference_diffusion_transformer_block(mod, a.clone(), s.clone(), z.clone()).float()
    dev = get_device(); ck = _ck(dev)
    def sub(p): return {k[len(p)+1:]: v for k, v in sd.items() if k.startswith(p + ".")}
    def s2(d, p): return {k[len(p)+1:]: v for k, v in d.items() if k.startswith(p + ".")}
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    ftt = lambda x: ttnn.from_torch(x.t(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    apb_sd = sub("attention_pair_bias")
    adaln_a = AdaLN(False, remap_adaptive_layernorm(s2(apb_sd, "layernorm_a")), ck)
    apb = AttentionPairBias(hd, n_heads, True, False, remap_attention_pair_bias(apb_sd), ck)
    lalw, lalb = ftt(apb_sd["linear_a_last.weight"]), ft(apb_sd["linear_a_last.bias"])
    ctb = sub("conditioned_transition_block")
    ctb_adaln = AdaLN(False, remap_adaptive_layernorm(s2(ctb, "adaln")), ck)
    a1w, a2w, bw = ftt(ctb["linear_nobias_a1.weight"]), ftt(ctb["linear_nobias_a2.weight"]), ftt(ctb["linear_nobias_b.weight"])
    lsw, lsb = ftt(ctb["linear_s.weight"]), ft(ctb["linear_s.bias"])
    a_t, s_t, z_t = ft(a), ft(s), ft(z)
    an = adaln_a(a_t, s_t)
    attn = apb(an, z_t)
    gate = ttnn.linear(s_t, lalw, bias=lalb, compute_kernel_config=ck, core_grid=CORE_GRID_MAIN)
    attn = ttnn.multiply(gate, attn, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
    cn = ctb_adaln(attn, s_t)
    c1 = ttnn.linear(cn, a1w, compute_kernel_config=ck, core_grid=CORE_GRID_MAIN)
    c2 = ttnn.linear(cn, a2w, compute_kernel_config=ck, core_grid=CORE_GRID_MAIN)
    cb = ttnn.multiply(c2, c1, input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
    cba = ttnn.linear(cb, bw, compute_kernel_config=ck, core_grid=CORE_GRID_MAIN)
    csg = ttnn.linear(s_t, lsw, bias=lsb, compute_kernel_config=ck, core_grid=CORE_GRID_MAIN)
    ff = ttnn.multiply(csg, cba, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
    out = ttnn.add(attn, ff)
    ot = torch.Tensor(ttnn.to_torch(out)).float().reshape(out_ref.shape)
    p = pcc(ot, out_ref)
    assert p > 0.98, f"PCC {p:.5f}"


# DistogramHead (pair -> distogram bins, symmetrized).
def test_distogram_head_parity():
    from tt_bio.esmfold2 import DistogramHead
    c_z, no_bins, L = 128, 64, 32
    mod, sd = make_distogram_head(c_z, no_bins, seed=0)
    z = torch.randn(1, L, L, c_z)
    ref = run_reference_distogram_head(mod, z.clone()).float()
    dev = get_device()
    dh = DistogramHead(remap_distogram_head(sd), _ck(dev))
    out = torch.Tensor(ttnn.to_torch(dh(ttnn.from_torch(z, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)))).float().reshape(ref.shape)
    p = pcc(out, ref)
    assert p > 0.98, f"PCC {p:.5f}"


# Real-weight validation against the public Protenix base checkpoint (v0.5.0).
# Skipped if the checkpoint isn't present. Strongest validation: confirms the
# weight remaps reproduce the reference on REAL trained weights, not just random.
_CKPT = "/home/ttuser/protenix_ckpt/protenix_base_default_v0.5.0.pt"


@pytest.mark.skipif(not os.path.exists(_CKPT), reason="protenix base checkpoint not downloaded")
def test_real_weight_pairformer_block():
    from protenix.model.modules.pairformer import PairformerBlock
    ck = torch.load(_CKPT, map_location="cpu", weights_only=False)["model"]
    pfx = "module.pairformer_stack.blocks.0."
    blk_sd = {k[len(pfx):]: v for k, v in ck.items() if k.startswith(pfx)}
    mod = PairformerBlock(n_heads=16, c_z=128, c_s=384, c_hidden_mul=128,
                          c_hidden_pair_att=32, no_heads_pair=4).eval()
    mod.load_state_dict(blk_sd, strict=True)  # real weights, exact load
    L = 64
    s = torch.randn(1, L, 384); z = torch.randn(1, L, L, 128)
    s_ref, z_ref = run_reference_pairformer_block(mod, s.clone(), z.clone())
    s_ref, z_ref = s_ref.float(), z_ref.float()
    dev = get_device()
    layer = PairformerLayer(32, 4, 24, 16, True,
                            remap_pairformer_block(mod.state_dict()), _ck(dev))
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    s_out, z_out = layer(ft(s), ft(z))
    so = torch.Tensor(ttnn.to_torch(s_out)).float().reshape(s_ref.shape)
    zo = torch.Tensor(ttnn.to_torch(z_out)).float().reshape(z_ref.shape)
    ps, pz = pcc(so, s_ref), pcc(zo, z_ref)
    assert ps > 0.98 and pz > 0.98, f"real-weight PCC s={ps:.5f} z={pz:.5f}"
