"""Tests for the ttnn ESMC implementation, against the PyTorch reference.

Idiom (matching tests/test_tenstorrent.py): build the reference with random
weights, load the *same* state_dict into the ttnn module, run both, compare.
Everything runs on TT device 0 (see tt_boltz.tenstorrent.get_device).
"""

import os
import sys

import pytest
import torch
import ttnn

sys.path.insert(0, os.path.dirname(__file__))
from esmc_reference import ESMC_300M, VOCAB_SIZE, make_esmc_300m  # noqa: E402

from tt_boltz.tenstorrent import WeightScope, get_device  # noqa: E402
from tt_boltz import esmc as tt_esmc  # noqa: E402

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
