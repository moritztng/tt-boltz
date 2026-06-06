"""Tests for the ttnn ESMC implementation, against the PyTorch reference.

Idiom (matching tests/test_tenstorrent.py): build the reference with random
weights, load the *same* state_dict into the ttnn module, run both, compare.
Everything runs on TT device 0 (see tt_bio.tenstorrent.get_device).
"""

import os
import sys

import pytest
import torch
import ttnn

sys.path.insert(0, os.path.dirname(__file__))
from esmc_reference import ESMC_300M, VOCAB_SIZE, make_esmc_300m  # noqa: E402

from tt_bio.tenstorrent import WeightScope, get_device  # noqa: E402
from tt_bio import esmc as tt_esmc  # noqa: E402

torch.set_grad_enabled(False)
torch.manual_seed(893)

D_MODEL = ESMC_300M["d_model"]


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _kernel_config():
    # Mirror TorchWrapper's compute kernel config selection.
    dev = get_device()
    cls = (
        ttnn.types.WormholeComputeKernelConfig
        if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig
    )
    return cls(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


@pytest.mark.parametrize("seq_len", [16, 128])
def test_embedding(seq_len):
    ref = make_esmc_300m()
    state = WeightScope.wrap(ref.state_dict()).child("embed").as_dict()

    dev = get_device()
    mod = tt_esmc.Embedding(state, _kernel_config())

    tokens = torch.randint(0, VOCAB_SIZE, (1, seq_len))
    ref_out = ref.embed(tokens)

    tokens_tt = ttnn.from_torch(
        tokens.to(torch.int32), device=dev, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32
    )
    out = torch.Tensor(ttnn.to_torch(mod(tokens_tt))).float()

    assert out.shape == ref_out.shape, (out.shape, ref_out.shape)
    p = pcc(out, ref_out)
    assert p > 0.999, f"PCC {p:.5f} too low"


@pytest.mark.parametrize("seq_len", [16, 128])
def test_rope(seq_len):
    from esm.layers.rotary import RotaryEmbedding

    n_heads, head_dim = ESMC_300M["n_heads"], D_MODEL // ESMC_300M["n_heads"]
    dev = get_device()

    # reference expects [B, L, H, D]; ttnn path uses [B, H, L, D].
    q = torch.randn(1, seq_len, n_heads, head_dim)
    k = torch.randn(1, seq_len, n_heads, head_dim)
    rotary = RotaryEmbedding(head_dim).eval()
    q_ref, k_ref = rotary(q.clone(), k.clone())  # [B, L, H, D]

    cos, sin = tt_esmc.rope_tables(seq_len, head_dim, device=dev)
    to_tt = lambda x: ttnn.from_torch(
        x.permute(0, 2, 1, 3).contiguous(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16
    )
    q_tt = torch.Tensor(ttnn.to_torch(tt_esmc.apply_rotary(to_tt(q), cos, sin))).float()
    k_tt = torch.Tensor(ttnn.to_torch(tt_esmc.apply_rotary(to_tt(k), cos, sin))).float()

    # bring ref to [B, H, L, D] to match
    for name, got, exp in [("q", q_tt, q_ref.permute(0, 2, 1, 3)), ("k", k_tt, k_ref.permute(0, 2, 1, 3))]:
        p = pcc(got, exp)
        assert p > 0.999, f"{name} PCC {p:.5f} too low"


@pytest.mark.parametrize("seq_len", [16, 128])
def test_attention(seq_len):
    n_heads, head_dim = ESMC_300M["n_heads"], D_MODEL // ESMC_300M["n_heads"]
    ref = make_esmc_300m()
    attn_ref = ref.transformer.blocks[0].attn
    state = WeightScope.wrap(ref.state_dict()).child("transformer.blocks.0.attn").as_dict()

    dev = get_device()
    mod = tt_esmc.Attention(n_heads, state, _kernel_config())

    x = torch.randn(1, seq_len, D_MODEL)
    ref_out, _ = attn_ref(x, None)

    cos, sin = tt_esmc.rope_tables(seq_len, head_dim, device=dev)
    x_tt = ttnn.from_torch(x, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    out = torch.Tensor(ttnn.to_torch(mod(x_tt, cos, sin))).float()

    assert out.shape == ref_out.shape, (out.shape, ref_out.shape)
    p = pcc(out, ref_out)
    assert p > 0.99, f"PCC {p:.5f} too low"


def test_attention_mask_isolates_segments():
    """The additive attn_mask (used for ESMC-6B chain-aware / padded attention)
    must block cross-segment attention: perturbing one segment's tokens leaves
    the other segment's attention output unchanged."""
    n_heads, head_dim = ESMC_300M["n_heads"], D_MODEL // ESMC_300M["n_heads"]
    ref = make_esmc_300m()
    state = WeightScope.wrap(ref.state_dict()).child("transformer.blocks.0.attn").as_dict()
    dev = get_device()
    mod = tt_esmc.Attention(n_heads, state, _kernel_config())

    seq_len, cut = 32, 16
    cos, sin = tt_esmc.rope_tables(seq_len, head_dim, device=dev)
    # block-diagonal mask: [0:cut] and [cut:] cannot attend to each other
    seg = torch.cat([torch.zeros(cut), torch.ones(seq_len - cut)]).long()
    allow = seg[:, None] == seg[None, :]
    m = torch.where(allow, 0.0, float("-inf"))[None, None]  # [1,1,L,L]
    m_tt = ttnn.from_torch(m.to(torch.bfloat16), device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    def run(x):
        x_tt = ttnn.from_torch(x, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        return torch.Tensor(ttnn.to_torch(mod(x_tt, cos, sin, m_tt))).float()

    x1 = torch.randn(1, seq_len, D_MODEL)
    x2 = x1.clone()
    x2[:, cut:] = torch.randn(1, seq_len - cut, D_MODEL)  # perturb second segment only
    o1, o2 = run(x1), run(x2)
    drift = (o1[:, :cut] - o2[:, :cut]).abs().max().item()
    assert drift < 1e-2, f"segment 0 leaked across mask (max drift {drift})"


def test_te_key_remap():
    """TransformerEngine 6B weight names map to the esm-repo nn.Sequential
    index names the ttnn blocks expect; _extra_state and the LM head drop out."""
    src = {
        "esmc.embed.weight": 0,
        "esmc.transformer.blocks.0.attn.layernorm_qkv.layer_norm_weight": 0,
        "esmc.transformer.blocks.0.attn.layernorm_qkv.layer_norm_bias": 0,
        "esmc.transformer.blocks.0.attn.layernorm_qkv.weight": 0,
        "esmc.transformer.blocks.0.attn.layernorm_qkv._extra_state": 0,
        "esmc.transformer.blocks.0.attn.q_ln.weight": 0,
        "esmc.transformer.blocks.0.attn.out_proj.weight": 0,
        "esmc.transformer.blocks.0.ffn.layer_norm_weight": 0,
        "esmc.transformer.blocks.0.ffn.fc1_weight": 0,
        "esmc.transformer.blocks.0.ffn.fc2_weight": 0,
        "esmc.transformer.norm.weight": 0,
        "lm_head.0.weight": 0,
    }
    remapped = set()
    for k in src:
        if k.endswith("_extra_state") or k.startswith("lm_head") or not k.startswith("esmc."):
            continue
        nk = k[len("esmc."):]
        for s, d in tt_esmc._TE_KEY_REMAP:
            nk = nk.replace(s, d)
        remapped.add(nk)
    assert "embed.weight" in remapped
    assert "transformer.blocks.0.attn.layernorm_qkv.0.weight" in remapped  # layer_norm_weight
    assert "transformer.blocks.0.attn.layernorm_qkv.0.bias" in remapped
    assert "transformer.blocks.0.attn.layernorm_qkv.1.weight" in remapped  # qkv proj
    assert "transformer.blocks.0.ffn.0.weight" in remapped  # ffn norm
    assert "transformer.blocks.0.ffn.1.weight" in remapped  # fc1
    assert "transformer.blocks.0.ffn.3.weight" in remapped  # fc2
    assert "transformer.norm.weight" in remapped
    assert not any("_extra_state" in k or "lm_head" in k for k in remapped)


def test_esmc_real_weights():
    """Validate against the trained ESMC-300M on a real protein (human ubiquitin).

    Skips unless the HF checkpoint is already cached (avoids a ~1GB download).
    """
    from huggingface_hub import try_to_load_from_cache

    _cfg, repo_id, wpath = tt_esmc.CONFIGS["esmc-300m"]
    cached = try_to_load_from_cache(repo_id, wpath)
    if not isinstance(cached, str):
        pytest.skip("ESMC-300M weights not cached; run ESMC.from_pretrained() first")

    sd = torch.load(cached, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    ref = make_esmc_300m()
    ref.load_state_dict(sd, strict=False)
    mod = tt_esmc.ESMC(**ESMC_300M)
    mod.load_state_dict(sd, strict=False)

    seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    tokens = tt_esmc.tokenize(seq)
    ref_logits, _ = ref(tokens)
    logits, _ = mod(tokens)

    # On trained weights the model is confident, so top-1 should match torch exactly.
    assert pcc(logits, ref_logits) > 0.999
    agree = (logits.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
    assert agree == 1.0, f"argmax agreement {agree:.3f} < 1.0"


@pytest.mark.parametrize("seq_len", [16, 64])
def test_esmc_end_to_end(seq_len):
    ref = make_esmc_300m()
    mod = tt_esmc.ESMC(**ESMC_300M)
    mod.load_state_dict(ref.state_dict(), strict=False)

    tokens = torch.randint(0, VOCAB_SIZE, (1, seq_len))
    ref_logits, ref_emb = ref(tokens)
    logits, emb = mod(tokens)

    assert logits.shape == ref_logits.shape, (logits.shape, ref_logits.shape)
    pe, pl = pcc(emb, ref_emb), pcc(logits, ref_logits)
    assert pe > 0.98, f"embeddings PCC {pe:.5f} too low"
    assert pl > 0.98, f"logits PCC {pl:.5f} too low"
    # argmax (predicted token) agreement is the practical end-to-end signal.
    agree = (logits.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
    assert agree > 0.9, f"argmax agreement {agree:.3f} too low"


@pytest.mark.parametrize("seq_len", [16, 128])
def test_block(seq_len):
    n_heads, head_dim = ESMC_300M["n_heads"], D_MODEL // ESMC_300M["n_heads"]
    ref = make_esmc_300m()
    block_ref = ref.transformer.blocks[0]
    state = WeightScope.wrap(ref.state_dict()).child("transformer.blocks.0").as_dict()

    dev = get_device()
    mod = tt_esmc.Block(n_heads, ESMC_300M["n_layers"], state, _kernel_config())

    x = torch.randn(1, seq_len, D_MODEL)
    ref_out, _ = block_ref(x, None, None, None, None)

    cos, sin = tt_esmc.rope_tables(seq_len, head_dim, device=dev)
    x_tt = ttnn.from_torch(x, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    out = torch.Tensor(ttnn.to_torch(mod(x_tt, cos, sin))).float()

    assert out.shape == ref_out.shape, (out.shape, ref_out.shape)
    p = pcc(out, ref_out)
    assert p > 0.99, f"PCC {p:.5f} too low"


@pytest.mark.parametrize("seq_len", [16, 128])
def test_ffn(seq_len):
    ref = make_esmc_300m()
    ffn_ref = ref.transformer.blocks[0].ffn
    state = WeightScope.wrap(ref.state_dict()).child("transformer.blocks.0.ffn").as_dict()

    dev = get_device()
    mod = tt_esmc.SwiGLUFFN(state, _kernel_config())

    x = torch.randn(1, seq_len, D_MODEL)
    ref_out = ffn_ref(x)

    x_tt = ttnn.from_torch(x, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    out = torch.Tensor(ttnn.to_torch(mod(x_tt))).float()

    assert out.shape == ref_out.shape, (out.shape, ref_out.shape)
    p = pcc(out, ref_out)
    assert p > 0.99, f"PCC {p:.5f} too low"
