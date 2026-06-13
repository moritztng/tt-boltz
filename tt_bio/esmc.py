"""ESMC protein language model on Tenstorrent (ttnn).

A from-scratch ttnn implementation of EvolutionaryScale / Biohub's ESMC
(Evolutionary Scale Modeling Cambrian) sequence-only protein language model,
built on the tt-bio ttnn framework (``tenstorrent.Module`` / ``WeightScope`` /
``get_device``). We start with the smallest variant, ESMC-300M.

Reference (PyTorch): ``/home/ttuser/esm`` — esm/models/esmc.py, esm/layers/*.
The reference forward (use_flash_attn=False) is:

    x = embed(tokens)                       # [B, L, d_model]
    x = transformer(x)                      # 30 x UnifiedTransformerBlock + final LayerNorm
    logits = sequence_head(x)               # [B, L, 64]

Built bottom-up, one tested component at a time. This module currently
implements: token embedding.
"""

from __future__ import annotations

import torch
import ttnn

from tt_bio.tenstorrent import (
    Module,
    TorchWrapper,
    Weights,
    WeightScope,
    _dtype,
    _sdpa_program_config_for_lengths,
    get_device,
)

VOCAB_SIZE = 64
ROPE_BASE = 10000.0

# Sequence vocab (esm.utils.constants.esm3.SEQUENCE_VOCAB): token id = index here.
SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T",
    "I", "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U",
    "Z", "O", ".", "-", "|", "<mask>",
]
BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, MASK_TOKEN = 0, 2, 3, 32
PAD_TOKEN = 1  # SEQUENCE_VOCAB index of <pad>
BUCKET = 64    # pad the LM length to a multiple of this to avoid per-length recompilation
_AA_TO_ID = {a: i for i, a in enumerate(SEQUENCE_VOCAB)}

# name -> (config, hf repo id, weights path within repo)
CONFIGS = {
    "esmc-300m": (
        dict(d_model=960, n_heads=15, n_layers=30),
        "biohub/esmc-300m-2024-12",
        "data/weights/esmc_300m_2024_12_v0.pth",
    ),
}

# Architecture configs for the larger variants. ESMC-6B is the LM backbone of
# ESMFold2; the ttnn ESMC architecture supports it via config (identical to
# 300M, just larger), validated by the 300M parity. Real-weight loading for 6B
# needs a sharded-safetensors + key-remap loader (transformers format, ~12GB)
# and block-fp8 to fit one Blackhole — separate from the single-.pth 300M path.
ARCH_CONFIGS = {
    "esmc-300m": dict(d_model=960, n_heads=15, n_layers=30),
    "esmc-600m": dict(d_model=1152, n_heads=18, n_layers=36),
    "esmc-6b": dict(d_model=2560, n_heads=40, n_layers=80),  # ESMFold2 LM backbone
}


def tokenize(sequence: str) -> "torch.Tensor":
    """Protein string -> token ids [1, L+2] with <cls>/<eos> (matches esm)."""
    ids = [BOS_TOKEN] + [_AA_TO_ID.get(c, UNK_TOKEN) for c in sequence.upper()] + [EOS_TOKEN]
    return torch.tensor([ids], dtype=torch.long)


def rope_tables(seq_len: int, head_dim: int, base: float = ROPE_BASE, device=None):
    """Precompute NeoX-style RoPE cos/sin tables, shaped [1, 1, L, head_dim].

    Mirrors esm.layers.rotary.RotaryEmbedding (scale_base=None, interleaved=False):
    inv_freq = 1 / base**(arange(0,d,2)/d); freqs = outer(arange(L), inv_freq);
    cos/sin duplicated along the last dim ([c0..c_{d/2-1}, c0..c_{d/2-1}]).
    """
    device = device or get_device()
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [L, d/2]
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).view(1, 1, seq_len, head_dim)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).view(1, 1, seq_len, head_dim)
    to_tt = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    return to_tt(cos), to_tt(sin)


def apply_rotary(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
    """Apply RoPE to x [B, H, L, head_dim]; cos/sin broadcast as [1, 1, L, head_dim].

    out = x * cos + rotate_half(x) * sin, rotate_half(x) = cat([-x2, x1]).
    """
    x1, x2 = ttnn.chunk(x, 2, dim=-1)
    rot = ttnn.concat([ttnn.neg(x2), x1], dim=-1)
    out = ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rot, sin))
    ttnn.deallocate(x1)
    ttnn.deallocate(x2)
    ttnn.deallocate(rot)
    return out


class Embedding(Module):
    """Token embedding lookup (mirrors nn.Embedding(64, d_model)).

    Weight key: ``<scope>.weight`` of shape [vocab, d_model] (no transpose).
    """

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        # Embedding table is indexed, not matmul'd: keep [vocab, d_model] as-is.
        self.weight = self.torch_to_tt("weight", transform=lambda x: x)

    def __call__(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        # tokens: ROW_MAJOR uint32 [B, L]; output [B, L, d_model] in TILE layout.
        return ttnn.embedding(
            tokens,
            self.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


class Attention(Module):
    """Multi-head self-attention with QK-LayerNorm + RoPE (no biases on projections).

    Mirrors esm.layers.attention.MultiHeadAttention (qk_layernorm=True, bias=False):
      qkv = Linear(LayerNorm(x)); q,k,v = chunk(qkv,3)
      q = LayerNorm(q); k = LayerNorm(k)            # over full d_model, then per-head RoPE
      o = SDPA(rope(q), rope(k), v, scale=d_head**-0.5); out_proj(o)
    """

    def __init__(self, n_heads: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.n_heads = n_heads
        self.in_norm_weight = self.torch_to_tt("layernorm_qkv.0.weight")
        self.in_norm_bias = self.torch_to_tt("layernorm_qkv.0.bias")
        # The two big projection weights (qkv, out_proj) carry the bulk of the
        # ESMC-6B's parameters; in fast mode they load as block-fp8 (bfloat8_b),
        # halving their weight-read bandwidth and resident size. _dtype() is bf16
        # otherwise (full precision, the default).
        self.qkv_weight = self.torch_to_tt("layernorm_qkv.1.weight", dtype=_dtype())
        self.q_ln_weight = self.torch_to_tt("q_ln.weight")
        self.k_ln_weight = self.torch_to_tt("k_ln.weight")
        self.out_weight = self.torch_to_tt("out_proj.weight", dtype=_dtype())

    def __call__(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor,
                 attn_mask: ttnn.Tensor | None = None,
                 key_valid: ttnn.Tensor | None = None) -> ttnn.Tensor:
        ck = self.compute_kernel_config
        d_model = x.shape[-1]
        head_dim = d_model // self.n_heads

        x_norm = ttnn.layer_norm(
            x, weight=self.in_norm_weight, bias=self.in_norm_bias,
            epsilon=1e-5, compute_kernel_config=ck,
        )
        qkv = self._lin(x_norm, self.qkv_weight)
        ttnn.deallocate(x_norm)

        q, k, v = ttnn.chunk(qkv, 3, dim=-1)
        ttnn.deallocate(qkv)
        q = ttnn.layer_norm(q, weight=self.q_ln_weight, epsilon=1e-5, compute_kernel_config=ck)
        k = ttnn.layer_norm(k, weight=self.k_ln_weight, epsilon=1e-5, compute_kernel_config=ck)

        # Re-pack and use the tile-aware head split, then apply per-head RoPE.
        qkv = ttnn.concat([q, k, v], dim=-1)
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        q, k, v = self._split_heads(qkv, self.n_heads)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        if key_valid is not None:
            # Zero padded keys/values so their attention contribution is exactly
            # 0 (weight x 0) — exact masking, not reliant on bf16 exp(-inf).
            k = ttnn.multiply(k, key_valid)
            v = ttnn.multiply(v, key_valid)

        o = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False, scale=head_dim**-0.5,
            program_config=_sdpa_program_config_for_lengths(q.shape[2], k.shape[2]),
        )
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        o = self._merge_heads(o)  # [B, L, d_model]
        out = self._lin(o, self.out_weight)
        ttnn.deallocate(o)
        return out


class SwiGLUFFN(Module):
    """SwiGLU feed-forward (mirrors esm.layers.blocks.swiglu_ln_ffn, bias=False):
      h = Linear(LayerNorm(x)); x1,x2 = chunk(h,2); Linear(silu(x1) * x2).
    """

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.norm_weight = self.torch_to_tt("0.weight")
        self.norm_bias = self.torch_to_tt("0.bias")
        # fc1/fc2 are the FFN's big matmuls (and the bulk of the ESMC-6B FLOPs);
        # block-fp8 in fast mode, bf16 otherwise. Shared with the folding trunk's
        # pair-transition, so fast mode bf8's that too.
        self.fc1_weight = self.torch_to_tt("1.weight", dtype=_dtype())
        self.fc2_weight = self.torch_to_tt("3.weight", dtype=_dtype())

    def _ffn(self, x: ttnn.Tensor) -> ttnn.Tensor:
        ck = self.compute_kernel_config
        x_norm = ttnn.layer_norm(
            x, weight=self.norm_weight, bias=self.norm_bias,
            epsilon=1e-5, compute_kernel_config=ck,
        )
        h = self._lin(x_norm, self.fc1_weight)
        ttnn.deallocate(x_norm)
        x1, x2 = ttnn.chunk(h, 2, dim=-1)
        ttnn.deallocate(h)
        gated = ttnn.multiply(ttnn.silu(x1), x2)
        ttnn.deallocate(x1); ttnn.deallocate(x2)
        out = self._lin(gated, self.fc2_weight)
        ttnn.deallocate(gated)
        return out

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # The fc1 activation (2*d_ff wide) is several GB at long L and, on top of
        # the resident 6B weights, OOMs the 12 GB/chip Wormhole DRAM. The FFN is
        # row-independent over dim=1, so tiling it is bit-exact. 4D pair input
        # (ESMFold2 MSA-encoder pair_transition, [B,L,L,c]) has transient ~ rows*L
        # -> area-bounded tile; 3D per-token (ESMC LM FFN, [B,L,d]) -> fixed row
        # tile. Single pass on Blackhole. See tenstorrent._apply_grid_thresholds.
        from tt_bio import tenstorrent
        L = x.shape[1]
        if len(x.shape) == 4:
            chunk = tenstorrent.pair_row_tile(L)
        else:
            t = tenstorrent.SMALL_GRID_SEQ_TILE
            chunk = t if (t and L > t) else 0
        if chunk:
            parts = ttnn.chunk(x, -(-L // chunk), dim=1)
            return ttnn.concat([self._ffn(p) for p in parts], dim=1)
        return self._ffn(x)


class Block(Module):
    """UnifiedTransformerBlock, plain path (mirrors esm.layers.blocks):
      x = x + attn(x) / s ; x = x + ffn(x) / s,  s = sqrt(n_layers / 36).
    """

    def __init__(self, n_heads: int, n_layers: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.attn = Attention(n_heads, self.scope("attn"), compute_kernel_config)
        self.ffn = SwiGLUFFN(self.scope("ffn"), compute_kernel_config)
        self.inv_scale = 1.0 / (n_layers / 36) ** 0.5

    def __call__(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor,
                 attn_mask: ttnn.Tensor | None = None,
                 key_valid: ttnn.Tensor | None = None) -> ttnn.Tensor:
        r1 = self.attn(x, cos, sin, attn_mask, key_valid)
        x = ttnn.add(x, ttnn.multiply(r1, self.inv_scale))
        ttnn.deallocate(r1)
        r3 = self.ffn(x)
        x = ttnn.add(x, ttnn.multiply(r3, self.inv_scale))
        ttnn.deallocate(r3)
        return x


class RegressionHead(Module):
    """Sequence head MLP (mirrors esm.layers.regression_head.RegressionHead, biases on):
      Linear -> GELU -> LayerNorm -> Linear.
    """

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        row = lambda x: x.reshape(1, -1)
        self.fc1_weight = self.torch_to_tt("0.weight")
        self.fc1_bias = self.torch_to_tt("0.bias", transform=row)
        self.norm_weight = self.torch_to_tt("2.weight")
        self.norm_bias = self.torch_to_tt("2.bias")
        self.fc2_weight = self.torch_to_tt("3.weight")
        self.fc2_bias = self.torch_to_tt("3.bias", transform=row)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        ck = self.compute_kernel_config
        a = self._lin(x, self.fc1_weight, bias=self.fc1_bias)
        a = ttnn.gelu(a)
        a = ttnn.layer_norm(
            a, weight=self.norm_weight, bias=self.norm_bias,
            epsilon=1e-5, compute_kernel_config=ck,
        )
        logits = self._lin(a, self.fc2_weight, bias=self.fc2_bias)
        ttnn.deallocate(a)
        return logits


class ESMCModel(Module):
    """Full ESMC stack: embed -> N blocks -> final LayerNorm (-> head).

    __call__ returns (logits[B,L,64], embeddings[B,L,d_model]); embeddings are
    the post-final-norm hidden states (matches esm.models.esmc.ESMC).
    """

    def __init__(self, n_heads: int, n_layers: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.n_heads = n_heads
        self.embed = Embedding(self.scope("embed"), compute_kernel_config)
        self.blocks = [
            Block(n_heads, n_layers, self.scope(f"transformer.blocks.{i}"), compute_kernel_config)
            for i in range(n_layers)
        ]
        self.norm_weight = self.torch_to_tt("transformer.norm.weight")
        self.head = RegressionHead(self.scope("sequence_head"), compute_kernel_config)

    def __call__(self, tokens: ttnn.Tensor):
        seq_len = tokens.shape[-1]
        head_dim = self.norm_weight.shape[-1] // self.n_heads
        cos, sin = rope_tables(seq_len, head_dim, device=self.device)

        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, cos, sin)
        emb = ttnn.layer_norm(
            x, weight=self.norm_weight, epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(x)
        logits = self.head(emb)
        return logits, emb


class ESMC(TorchWrapper):
    """Top-level ESMC model (torch in / torch out). Mirrors esm.models.esmc.ESMC.

    Usage: m = ESMC(d_model, n_heads, n_layers); m.load_state_dict(sd); m(tokens).
    forward(tokens[int B,L]) -> (logits[B,L,64], embeddings[B,L,d_model]).
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

    @classmethod
    def from_pretrained(cls, name: str = "esmc-300m") -> "ESMC":
        """Download + load trained weights from HuggingFace (e.g. 'esmc-300m')."""
        from huggingface_hub import hf_hub_download

        config, repo_id, weights_path = CONFIGS[name]
        path = hf_hub_download(repo_id, weights_path)
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        model = cls(**config)
        model.load_state_dict(sd, strict=False)
        return model

    def _create_module(self, weights: WeightScope) -> ESMCModel:
        return ESMCModel(self.n_heads, self.n_layers, weights, self.compute_kernel_config)

    def forward(self, tokens: torch.Tensor):
        tokens_tt = ttnn.from_torch(
            tokens.to(torch.int32), device=self.tt_device,
            layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        )
        logits, emb = self.module(tokens_tt)
        return self._to_torch(logits), self._to_torch(emb)


# ===========================================================================
# ESMC-6B language-model backbone for ESMFold2
# ===========================================================================
#
# The 6B checkpoint ships in HuggingFace transformers / TransformerEngine
# layout (sharded safetensors, fused LayerNormLinear / LayerNormMLP modules),
# so its weight keys differ from the esm-repo names the ttnn blocks expect.
# This remap renames TE keys to the esm-repo `nn.Sequential`-index names, after
# which the existing `Block` / `Embedding` modules load unchanged.

_TE_KEY_REMAP = (
    ("attn.layernorm_qkv.layer_norm_weight", "attn.layernorm_qkv.0.weight"),
    ("attn.layernorm_qkv.layer_norm_bias", "attn.layernorm_qkv.0.bias"),
    ("attn.layernorm_qkv.weight", "attn.layernorm_qkv.1.weight"),
    ("ffn.layer_norm_weight", "ffn.0.weight"),
    ("ffn.layer_norm_bias", "ffn.0.bias"),
    ("ffn.fc1_weight", "ffn.1.weight"),
    ("ffn.fc2_weight", "ffn.3.weight"),
)


def load_esmc6b_state_dict(snapshot_dir: str) -> dict:
    """Read the sharded 6B safetensors and remap TE keys to esm-repo names.

    Keeps only weights the ttnn stack consumes (embed, transformer blocks,
    final norm); drops `_extra_state`, the LM head and any classifier heads.
    """
    import glob
    import json
    import os

    from safetensors import safe_open

    import tt_bio.tenstorrent as _tt

    # Load straight to bf16 (the device dtype) so the upload moves/tiles half the
    # data — ~2.6x faster ESMC-6B load, bit-identical (fp32->bf16 rounding just
    # happens once, here vs in from_torch). In fast mode the big matmul weights
    # become block-fp8, whose quantization is sensitive to the fp32 mantissa, so
    # keep fp32 there to preserve exact fast-mode numerics.
    load_dtype = torch.float32 if _tt._FAST_MODE else torch.bfloat16
    idx_path = os.path.join(snapshot_dir, "model.safetensors.index.json")
    weight_map = json.load(open(idx_path))["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for k, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(k)

    sd: dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(snapshot_dir, shard), "pt") as f:
            for k in keys:
                if k.endswith("_extra_state") or k.startswith("lm_head"):
                    continue
                if not k.startswith("esmc."):
                    continue
                nk = k[len("esmc."):]  # drop the "esmc." prefix
                for src, dst in _TE_KEY_REMAP:
                    nk = nk.replace(src, dst)
                sd[nk] = f.get_tensor(k).to(load_dtype)
    _ = glob  # (kept for symmetry with other loaders)
    return sd


class ESMCHiddenStatesModel(Module):
    """ESMC stack returning all `n_layers + 1` hidden states (ESMFold2 LM input).

    Matches `EsmcTransformerStack` collection semantics:
    `hs[0]` = embedding output, `hs[i]` = input to block `i` (= output of block
    `i-1`) for `1 <= i < n_layers`, and `hs[n_layers]` = final-LayerNorm output.
    Single-sequence / single-chain only (full attention, no padding) — which is
    how `compute_lm_hidden_states` feeds one wrapped chain at a time.
    """

    def __init__(self, n_heads: int, n_layers: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.embed = Embedding(self.scope("embed"), compute_kernel_config)
        self.blocks = [
            Block(n_heads, n_layers, self.scope(f"transformer.blocks.{i}"), compute_kernel_config)
            for i in range(n_layers)
        ]
        self.norm_weight = self.torch_to_tt("transformer.norm.weight")

    def __call__(self, tokens: ttnn.Tensor, attn_mask: ttnn.Tensor | None = None,
                 key_valid: ttnn.Tensor | None = None):
        seq_len = tokens.shape[-1]
        head_dim = self.norm_weight.shape[-1] // self.n_heads
        cos, sin = rope_tables(seq_len, head_dim, device=self.device)

        x = self.embed(tokens)
        hidden = [self._to_host(x)]  # hs[0] = embedding output
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, attn_mask, key_valid)
            if i < self.n_layers - 1:
                hidden.append(self._to_host(x))  # hs[i+1] = block i output
        norm_x = ttnn.layer_norm(
            x, weight=self.norm_weight, epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(x)
        hidden.append(self._to_host(norm_x))  # hs[n_layers] = post-norm output
        return hidden

    @staticmethod
    def _to_host(t: ttnn.Tensor) -> torch.Tensor:
        return torch.Tensor(ttnn.to_torch(t)).float()


class ESMCLanguageModel(TorchWrapper):
    """ESMC-6B backbone (torch in / torch out) producing ESMFold2 LM hidden states.

    `forward(input_ids[B,L])` -> hidden states `[n_layers+1, B, L, d_model]`,
    matching `transformers` ESMC `output_hidden_states=True` (the stacked input
    consumed by ESMFold2's `LanguageModelShim`).
    """

    def __init__(self, name: str = "esmc-6b"):
        super().__init__()
        cfg = ARCH_CONFIGS[name]
        self.d_model = cfg["d_model"]
        self.n_heads = cfg["n_heads"]
        self.n_layers = cfg["n_layers"]

    @classmethod
    def from_pretrained(cls, repo_id: str = "biohub/ESMC-6B", name: str = "esmc-6b") -> "ESMCLanguageModel":
        from huggingface_hub import snapshot_download

        snap = snapshot_download(repo_id)
        model = cls(name=name)
        model.load_state_dict(load_esmc6b_state_dict(snap), strict=False)
        return model

    def _create_module(self, weights: WeightScope) -> ESMCHiddenStatesModel:
        return ESMCHiddenStatesModel(self.n_heads, self.n_layers, weights, self.compute_kernel_config)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, Lm = input_ids.shape
        # Bucket the LM length to a multiple of 64 so the 80-layer ESMC kernels
        # are shared across nearby lengths instead of recompiling per length.
        # Padded tokens are masked out of attention (additive -inf, seq_id-style
        # mask like the reference) and sliced off — the residual numerical effect
        # is within the diffusion's seed-to-seed noise floor.
        Lb = ((Lm + BUCKET - 1) // BUCKET) * BUCKET
        if Lb != Lm:
            pad = Lb - Lm
            input_ids = torch.nn.functional.pad(input_ids, (0, pad), value=PAD_TOKEN)
            if attn_mask is None:
                attn_mask = torch.zeros(B, Lb, Lb, dtype=torch.float32)
            else:
                attn_mask = torch.nn.functional.pad(attn_mask, (0, pad, 0, pad), value=0.0)
            attn_mask[:, :, Lm:] = float("-inf")  # no token attends to padded keys
        tokens_tt = ttnn.from_torch(
            input_ids.to(torch.int32), device=self.tt_device,
            layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        )
        mask_tt = key_valid_tt = None
        if attn_mask is not None:
            # [B,L,L] additive mask -> [B,1,L,L] bf16 for SDPA
            mask_tt = ttnn.from_torch(
                attn_mask.unsqueeze(1).to(torch.bfloat16), device=self.tt_device,
                layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
            )
        if Lb != Lm:
            kv = torch.ones(1, 1, Lb, 1); kv[:, :, Lm:, :] = 0.0  # zero padded keys/values
            key_valid_tt = ttnn.from_torch(
                kv.to(torch.bfloat16), device=self.tt_device,
                layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
            )
        hidden = self.module(tokens_tt, mask_tt, key_valid_tt)  # list of [B, Lb, d_model]
        return torch.stack(hidden, dim=0)[:, :, :Lm, :]  # slice padding -> [n+1, B, Lm, d_model]

    def release(self):
        """Free all ttnn device weights (≈12.8 GB for the 6B). Call after the
        single LM forward so the folding trunk reclaims DRAM on long sequences.
        Hidden states are already on host, so only weights are released."""
        if self.module is not None:
            _free_ttnn_tensors(self.module)
            self.module = None


def _free_ttnn_tensors(obj, seen=None):
    """Recursively ttnn.deallocate every device tensor reachable from `obj`."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, ttnn.Tensor):
        try:
            ttnn.deallocate(obj)
        except Exception:
            pass
        return
    if isinstance(obj, (list, tuple, set)):
        for x in obj:
            _free_ttnn_tensors(x, seen)
        return
    if isinstance(obj, dict):
        for x in obj.values():
            _free_ttnn_tensors(x, seen)
        return
    d = getattr(obj, "__dict__", None)
    if d:
        for x in list(d.values()):
            _free_ttnn_tensors(x, seen)
