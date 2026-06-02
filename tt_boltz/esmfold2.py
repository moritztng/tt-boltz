"""ESMFold2 on Tenstorrent (ttnn) — folding trunk.

ESMFold2 is a diffusion all-atom structure predictor built on ESMC-6B
embeddings. Reference architecture: the Biohub fork of transformers
(models/esmfold2/modeling_esmfold2_common.py), see tests/esmfold2_reference.py.

This module implements the `folding_trunk`: 48 PairUpdateBlocks, each
    z = z + tri_mul_out(z); z = z + tri_mul_in(z); z = z + pair_transition(z)
operating on the pair representation z [B, L, L, c_z=256].

The triangle-multiplication math is identical to AlphaFold3 / Boltz, so we
reuse tt-boltz's proven ttnn `TriangleMultiplication` kernel and the ESMC
`SwiGLUFFN`, remapping ESMFold2's bundled weights into the expected layout:
  proj_bundle[:512]->p_in, proj_bundle[512:]->g_in, proj_emit->p_out,
  proj_gate->g_out, norm_start->norm_in, norm_mix->norm_out.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import ttnn

from tt_boltz.esmc import SwiGLUFFN, apply_rotary
from tt_boltz.tenstorrent import (
    CORE_GRID_MAIN,
    Module,
    TorchWrapper,
    TriangleMultiplication,
    Weights,
    WeightScope,
    _sdpa_program_config_for_lengths,
)

_ROW = lambda x: x.reshape(1, -1)

C_Z = 256  # pair channels
PAD_MULTIPLE = 32  # pad seq to a tile multiple so the triangle contraction excludes padding


def _remap_trimul(sd: dict, prefix: str) -> WeightScope:
    """ESMFold2 TriangleMultiplicativeBlock weights -> tt-boltz TriangleMultiplication."""
    g = lambda k: sd[f"{prefix}.{k}"]
    pb = g("proj_bundle.weight")  # [4*latent, c_z]; rows: signal(2*latent) | gate(2*latent)
    half = pb.shape[0] // 2
    return WeightScope({
        "norm_in.weight": g("norm_start.weight"),
        "norm_in.bias": g("norm_start.bias"),
        "norm_out.weight": g("norm_mix.weight"),
        "norm_out.bias": g("norm_mix.bias"),
        "p_in.weight": pb[:half],   # signal (left|right values)
        "g_in.weight": pb[half:],   # gate (left|right gates)
        "p_out.weight": g("proj_emit.weight"),
        "g_out.weight": g("proj_gate.weight"),
    })


def _remap_transition(sd: dict) -> WeightScope:
    """ESMFold2 pair_transition (LN + SwiGLUMLP w12/w3) -> ESMC SwiGLUFFN keys."""
    return WeightScope({
        "0.weight": sd["pair_transition.norm.weight"],
        "0.bias": sd["pair_transition.norm.bias"],
        "1.weight": sd["pair_transition.ffn.w12.weight"],
        "3.weight": sd["pair_transition.ffn.w3.weight"],
    })


class PairUpdateBlock(Module):
    """z = z + tri_mul_out(z); z = z + tri_mul_in(z); z = z + pair_transition(z)."""

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        sd = self.weights.as_dict()
        # tri_mul_out = outgoing (ending=False); tri_mul_in = incoming (ending=True).
        self.tri_out = TriangleMultiplication(
            False, _remap_trimul(sd, "tri_mul_out._engine"), compute_kernel_config
        )
        self.tri_in = TriangleMultiplication(
            True, _remap_trimul(sd, "tri_mul_in._engine"), compute_kernel_config
        )
        self.transition = SwiGLUFFN(_remap_transition(sd), compute_kernel_config)

    def _residual(self, z: ttnn.Tensor, update: ttnn.Tensor) -> ttnn.Tensor:
        out = ttnn.add(z, update)
        ttnn.deallocate(z)
        ttnn.deallocate(update)
        return out

    def __call__(self, z: ttnn.Tensor, mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        z = self._residual(z, self.tri_out(z, mask))
        z = self._residual(z, self.tri_in(z, mask))
        z = self._residual(z, self.transition(z))
        return z


class FoldingTrunkModel(Module):
    """ModuleList of PairUpdateBlocks over the pair representation z."""

    def __init__(self, n_layers: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.blocks = [
            PairUpdateBlock(self.scope(f"blocks.{i}"), compute_kernel_config)
            for i in range(n_layers)
        ]

    def __call__(self, z: ttnn.Tensor, mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        for block in self.blocks:
            z = block(z, mask)
        return z


class FoldingTrunk(TorchWrapper):
    """Top-level folding trunk (torch z [B,L,L,256] in/out)."""

    def __init__(self, n_layers: int = 48, d_pair: int = C_Z):
        super().__init__()
        self.n_layers = n_layers
        self.d_pair = d_pair

    def _create_module(self, weights: WeightScope) -> FoldingTrunkModel:
        return FoldingTrunkModel(self.n_layers, weights, self.compute_kernel_config)

    def forward(self, z):
        # Pad both sequence axes to a tile multiple and mask padding out of the
        # triangle contraction, then slice back (mirrors tt-boltz PairformerModule).
        seq_len = z.shape[1]
        pad = (-seq_len) % PAD_MULTIPLE
        mask = None
        if pad:
            z = F.pad(z, (0, 0, 0, pad, 0, pad))
            real = torch.zeros(1, z.shape[1], z.shape[1])
            real[:, :seq_len, :seq_len] = 1.0
            mask = self._from_torch(real)
        out = self.module(self._from_torch(z), mask)
        return self._to_torch(out)[:, :seq_len, :seq_len, :]


# ===========================================================================
# Diffusion structure head — token transformer (DiT with pair bias)
# ===========================================================================


class AdaLN(Module):
    """Adaptive LayerNorm (adaLN-Zero): sigmoid(s_gate(LN_s(s)))*LN(a) + s_shift(LN_s(s))."""

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.s_scale = self.torch_to_tt("s_scale", transform=lambda x: x)
        self.s_gate_w = self.torch_to_tt("s_gate.weight")
        self.s_gate_b = self.torch_to_tt("s_gate.bias", transform=_ROW)
        self.s_shift_w = self.torch_to_tt("s_shift.weight")

    def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor) -> ttnn.Tensor:
        ck = self.compute_kernel_config
        a_norm = ttnn.layer_norm(a, epsilon=1e-5, compute_kernel_config=ck)
        s_norm = ttnn.layer_norm(s, weight=self.s_scale, epsilon=1e-5, compute_kernel_config=ck)
        gate = ttnn.sigmoid(ttnn.linear(
            s_norm, self.s_gate_w, bias=self.s_gate_b, compute_kernel_config=ck,
            dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN,
        ))
        shift = ttnn.linear(
            s_norm, self.s_shift_w, compute_kernel_config=ck,
            dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN,
        )
        return ttnn.add(ttnn.multiply(gate, a_norm), shift)


class AttentionPairBias(Module):
    """Gated multi-head attention conditioned on s (adaLN) with per-head pair bias from z."""

    def __init__(self, num_heads: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.num_heads = num_heads
        self.adaln = AdaLN(self.scope("adaln"), compute_kernel_config)
        self.q_w = self.torch_to_tt("q_proj.weight")
        self.q_b = self.torch_to_tt("q_proj.bias", transform=_ROW)
        self.kv_w = self.torch_to_tt("kv_proj.weight")
        self.g_w = self.torch_to_tt("g_proj.weight")
        self.out_gate_w = self.torch_to_tt("out_gate.weight")
        self.out_gate_b = self.torch_to_tt("out_gate.bias", transform=_ROW)
        self.pair_norm_w = self.torch_to_tt("pair_norm.weight")
        self.pair_norm_b = self.torch_to_tt("pair_norm.bias")
        self.pair_bias_w = self.torch_to_tt("pair_bias_proj.weight")

        # head_dim may not be a tile multiple (e.g. 768/16=48 -> padded to 64 by
        # the nlp head ops). Fold the head un-padding into two constant scatter
        # matmuls: pad g (d_model->Dpad) and a padded out_proj (Dpad->d_model).
        d_model = self.weights["out_proj.weight"].shape[0]
        H, hd = num_heads, d_model // num_heads
        self.head_dim = hd
        hdp = -(-hd // 32) * 32  # round up to tile multiple
        self.head_dim_pad = hdp
        Dpad = H * hdp
        Sg = torch.zeros(d_model, Dpad)  # g[.,768] @ Sg -> g[.,1024] (real slots only)
        for h in range(H):
            Sg[h * hd : (h + 1) * hd, h * hdp : h * hdp + hd] = torch.eye(hd)
        owt = self.weights["out_proj.weight"].t()  # [in d_model, out d_model]
        Op = torch.zeros(Dpad, d_model)  # padded out_proj input dim
        for h in range(H):
            Op[h * hdp : h * hdp + hd, :] = owt[h * hd : (h + 1) * hd, :]
        to_tt = lambda x: ttnn.from_torch(
            x, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16
        )
        self.scatter_g = to_tt(Sg)
        self.out_w_padded = to_tt(Op)

    def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor, z: ttnn.Tensor) -> ttnn.Tensor:
        ck = self.compute_kernel_config
        d_model = a.shape[-1]
        head_dim = d_model // self.num_heads

        x = self.adaln(a, s)
        lin = lambda t, w, b=None: ttnn.linear(
            t, w, bias=b, compute_kernel_config=ck, dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN
        )
        q = lin(x, self.q_w, self.q_b)
        kv = lin(x, self.kv_w)
        k, v = ttnn.chunk(kv, 2, dim=-1)
        ttnn.deallocate(kv)
        g = ttnn.sigmoid(lin(x, self.g_w))
        ttnn.deallocate(x)

        # Pack q,k,v and split into heads (tile-aware), then SDPA with pair bias.
        qkv = ttnn.concat([q, k, v], dim=-1)
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        qkv = ttnn.unsqueeze(qkv, 1)
        qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=self.num_heads, num_kv_heads=self.num_heads,
            transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)

        z_norm = ttnn.layer_norm(
            z, weight=self.pair_norm_w, bias=self.pair_norm_b, epsilon=1e-5, compute_kernel_config=ck
        )
        pair_bias = lin(z_norm, self.pair_bias_w)  # [B, L, L, H]
        ttnn.deallocate(z_norm)
        pair_bias = ttnn.permute(pair_bias, (0, 3, 1, 2))  # [B, H, L, L]

        ctx = ttnn.transformer.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=pair_bias, is_causal=False, scale=head_dim**-0.5,
            program_config=_sdpa_program_config_for_lengths(qh.shape[2], kh.shape[2]),
        )
        ttnn.deallocate(qh); ttnn.deallocate(kh); ttnn.deallocate(vh)
        ttnn.deallocate(pair_bias)
        ctx = ttnn.experimental.nlp_concat_heads(ctx, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ctx = ttnn.squeeze(ctx, 1)  # [B, L, H*head_dim_pad]  (pad value dims are 0)
        # Gate + out_proj in the padded head layout, then project back to d_model.
        g_pad = lin(g, self.scatter_g)  # [B, L, Dpad]
        ttnn.deallocate(g)
        ctx = ttnn.multiply(ctx, g_pad)
        ttnn.deallocate(g_pad)
        out = lin(ctx, self.out_w_padded)  # [B, L, d_model]
        ttnn.deallocate(ctx)
        out_gate = ttnn.sigmoid(lin(s, self.out_gate_w, self.out_gate_b))
        return ttnn.multiply(out_gate, out)


class ConditionedTransitionBlock(Module):
    """adaLN-conditioned SwiGLU transition, gated by sigmoid(output_gate(s))."""

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.adaln = AdaLN(self.scope("adaln"), compute_kernel_config)
        self.swish_w = self.torch_to_tt("lin_swish.weight")
        self.out_w = self.torch_to_tt("lin_out.weight")
        self.gate_w = self.torch_to_tt("output_gate.weight")
        self.gate_b = self.torch_to_tt("output_gate.bias", transform=_ROW)

    def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor) -> ttnn.Tensor:
        ck = self.compute_kernel_config
        lin = lambda t, w, b=None: ttnn.linear(
            t, w, bias=b, compute_kernel_config=ck, dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN
        )
        x = self.adaln(a, s)
        sw = lin(x, self.swish_w)
        ttnn.deallocate(x)
        a1, a2 = ttnn.chunk(sw, 2, dim=-1)
        ttnn.deallocate(sw)
        b = ttnn.multiply(ttnn.silu(a1), a2)
        ttnn.deallocate(a1); ttnn.deallocate(a2)
        out = lin(b, self.out_w)
        ttnn.deallocate(b)
        gate = ttnn.sigmoid(lin(s, self.gate_w, self.gate_b))
        return ttnn.multiply(gate, out)


class DiffusionTransformerModel(Module):
    """Token DiT: x = x + attn(x,s,z); x = x + transition(x,s), repeated."""

    def __init__(self, num_heads: int, num_blocks: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.attn = [
            AttentionPairBias(num_heads, self.scope(f"attn_blocks.{i}"), compute_kernel_config)
            for i in range(num_blocks)
        ]
        self.trans = [
            ConditionedTransitionBlock(self.scope(f"transition_blocks.{i}"), compute_kernel_config)
            for i in range(num_blocks)
        ]

    def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor, z: ttnn.Tensor) -> ttnn.Tensor:
        x = a
        for attn, trans in zip(self.attn, self.trans):
            x = ttnn.add(x, attn(x, s, z))
            x = ttnn.add(x, trans(x, s))
        return x


class DiffusionTransformer(TorchWrapper):
    """Top-level token transformer (torch a[B,L,768], s[B,L,768], z[B,L,L,256] in/out)."""

    def __init__(self, num_heads: int = 16, num_blocks: int = 12):
        super().__init__()
        self.num_heads = num_heads
        self.num_blocks = num_blocks

    def _create_module(self, weights: WeightScope) -> DiffusionTransformerModel:
        return DiffusionTransformerModel(
            self.num_heads, self.num_blocks, weights, self.compute_kernel_config
        )

    def forward(self, a, s, z):
        out = self.module(self._from_torch(a), self._from_torch(s), self._from_torch(z))
        return self._to_torch(out)


# ===========================================================================
# Diffusion conditioning (noise + s/z projections)
# ===========================================================================


class TransitionLayer(Module):
    """SwiGLU transition with separate a/b projections: out_proj(silu(a(LN(x)))*b(LN(x)))."""

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.norm_w = self.torch_to_tt("norm.weight")
        self.norm_b = self.torch_to_tt("norm.bias")
        self.a_w = self.torch_to_tt("a_proj.weight")
        self.b_w = self.torch_to_tt("b_proj.weight")
        self.out_w = self.torch_to_tt("out_proj.weight")

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        ck = self.compute_kernel_config
        lin = lambda t, w: ttnn.linear(
            t, w, compute_kernel_config=ck, dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN
        )
        xn = ttnn.layer_norm(x, weight=self.norm_w, bias=self.norm_b, epsilon=1e-5, compute_kernel_config=ck)
        a = lin(xn, self.a_w)
        b = lin(xn, self.b_w)
        ttnn.deallocate(xn)
        gated = ttnn.multiply(ttnn.silu(a), b)
        ttnn.deallocate(a); ttnn.deallocate(b)
        out = lin(gated, self.out_w)
        ttnn.deallocate(gated)
        return out


class DiffusionConditioningModel(Module):
    """Builds conditioning single s [B,L,c_s] and pair z [B,L,L,c_z] from inputs + noise."""

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.z_in_w = self.torch_to_tt("z_input_norm.weight")
        self.z_in_b = self.torch_to_tt("z_input_norm.bias")
        self.z_proj_w = self.torch_to_tt("z_proj.weight")
        self.z_trans = [TransitionLayer(self.scope(f"z_transitions.{i}"), compute_kernel_config) for i in range(2)]
        self.s_in_w = self.torch_to_tt("s_input_norm.weight")
        self.s_in_b = self.torch_to_tt("s_input_norm.bias")
        self.s_proj_w = self.torch_to_tt("s_proj.weight")
        self.noise_n_w = self.torch_to_tt("noise_norm.weight")
        self.noise_n_b = self.torch_to_tt("noise_norm.bias")
        self.noise_proj_w = self.torch_to_tt("noise_proj.weight")
        self.s_trans = [TransitionLayer(self.scope(f"s_transitions.{i}"), compute_kernel_config) for i in range(2)]

    def __call__(self, s_inputs, z_trunk, relpos, n_raw):
        ck = self.compute_kernel_config
        lin = lambda t, w: ttnn.linear(
            t, w, compute_kernel_config=ck, dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN
        )
        z = ttnn.concat([z_trunk, relpos], dim=-1)
        z = lin(ttnn.layer_norm(z, weight=self.z_in_w, bias=self.z_in_b, epsilon=1e-5, compute_kernel_config=ck), self.z_proj_w)
        for t in self.z_trans:
            z = ttnn.add(z, t(z))

        s = lin(ttnn.layer_norm(s_inputs, weight=self.s_in_w, bias=self.s_in_b, epsilon=1e-5, compute_kernel_config=ck), self.s_proj_w)
        n = lin(ttnn.layer_norm(n_raw, weight=self.noise_n_w, bias=self.noise_n_b, epsilon=1e-5, compute_kernel_config=ck), self.noise_proj_w)
        s = ttnn.add(s, ttnn.unsqueeze(n, 1))  # broadcast noise over the L axis
        for t in self.s_trans:
            s = ttnn.add(s, t(s))
        return s, z


class DiffusionConditioning(TorchWrapper):
    """Top-level conditioning (torch in/out). forward(t_hat, s_inputs, z_trunk, relpos) -> (s, z)."""

    def __init__(self, sigma_data: float = 16.0):
        super().__init__()
        self.sigma_data = sigma_data

    def _create_module(self, weights: WeightScope) -> DiffusionConditioningModel:
        self._fw = weights["fourier.w"]  # [fourier_dim] buffers (host fourier embed)
        self._fb = weights["fourier.b"]
        return DiffusionConditioningModel(weights, self.compute_kernel_config)

    def forward(self, t_hat, s_inputs, z_trunk, relpos):
        bsz = z_trunk.shape[0]
        t = torch.as_tensor(t_hat, dtype=torch.float32).reshape(-1)
        if t.numel() == 1:
            t = t.expand(bsz)
        t_noise = 0.25 * torch.log((t / self.sigma_data).clamp(min=1e-20))
        n_raw = torch.cos(2.0 * math.pi * (t_noise[:, None] * self._fw[None, :] + self._fb[None, :]))
        s, z = self.module(
            self._from_torch(s_inputs), self._from_torch(z_trunk),
            self._from_torch(relpos), self._from_torch(n_raw.float()),
        )
        return self._to_torch(s), self._to_torch(z)


# ===========================================================================
# Atom encoder/decoder — SWA (sliding-window) attention with 3D RoPE
# ===========================================================================


def build_3d_rope_tables(ref_pos, ref_space_uid, head_dim, n_spatial_per_axis,
                         n_uid_pairs, spatial_base_freq, uid_base_freq):
    """Host-side 3D RoPE cos/sin, returned duplicated as [B, 1, N, head_dim]
    (matches modeling_esmfold2_common.build_3d_rope + apply_rotary_emb_3d)."""
    B, N = ref_pos.shape[:2]
    half = head_dim // 2
    spatial_inv = 1.0 / (spatial_base_freq ** (torch.arange(n_spatial_per_axis, dtype=torch.float32) / n_spatial_per_axis))
    uid_inv = 1.0 / (uid_base_freq ** (torch.arange(n_uid_pairs, dtype=torch.float32) / n_uid_pairs))
    spatial = torch.einsum("bna,k->bnak", ref_pos.float(), spatial_inv).reshape(B, N, 3 * n_spatial_per_axis)
    uid = torch.einsum("bn,k->bnk", ref_space_uid.float(), uid_inv)
    freqs = torch.cat([spatial, uid], dim=-1)
    if freqs.shape[-1] < half:
        freqs = torch.cat([freqs, torch.zeros(B, N, half - freqs.shape[-1])], dim=-1)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).view(B, 1, N, head_dim)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).view(B, 1, N, head_dim)
    return cos, sin


def band_mask(n: int, half_window: int):
    """Sliding-window additive attention bias [1, 1, N, N] (0 inside band, -inf outside)."""
    idx = torch.arange(n)
    allowed = (idx[:, None] - idx[None, :]).abs() <= half_window
    return torch.where(allowed, 0.0, float("-inf")).view(1, 1, n, n)


def _rms_norm(x, ck, eps=1e-6):
    return ttnn.rms_norm(x, epsilon=eps, compute_kernel_config=ck)


class SWAAttention(Module):
    """Sliding-window attention with 3D RoPE, qk RMSNorm, sigmoid gate."""

    def __init__(self, n_heads: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.n_heads = n_heads
        self.qkv_w = self.torch_to_tt("Wqkv.weight")
        self.gate_w = self.torch_to_tt("gate_proj.weight")
        self.out_w = self.torch_to_tt("out_proj.weight")

    def __call__(self, x, cos, sin, attn_mask):
        ck = self.compute_kernel_config
        lin = lambda t, w: ttnn.linear(t, w, compute_kernel_config=ck, dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)
        head_dim = x.shape[-1] // self.n_heads
        qkv = ttnn.unsqueeze(lin(x, self.qkv_w), 1)  # [B,1,N,3d]
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=self.n_heads, num_kv_heads=self.n_heads,
            transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        q = apply_rotary(_rms_norm(q, ck), cos, sin)
        k = apply_rotary(_rms_norm(k, ck), cos, sin)
        ctx = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False, scale=head_dim**-0.5,
            program_config=_sdpa_program_config_for_lengths(q.shape[2], k.shape[2]),
        )
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        ctx = ttnn.squeeze(ttnn.experimental.nlp_concat_heads(ctx, memory_config=ttnn.DRAM_MEMORY_CONFIG), 1)
        gate = ttnn.sigmoid(lin(x, self.gate_w))
        ctx = ttnn.multiply(ctx, gate)
        ttnn.deallocate(gate)
        out = lin(ctx, self.out_w)
        ttnn.deallocate(ctx)
        return out


class SWAAtomBlock(Module):
    """adaLN-Zero (RMSNorm * (1+scale) + shift) + SWA attn + SwiGLU FFN, gated residuals."""

    def __init__(self, n_heads: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.adaln_w = self.torch_to_tt("adaln_modulation.1.weight")
        self.attn = SWAAttention(n_heads, self.scope("attn"), compute_kernel_config)
        self.ffn_up = self.torch_to_tt("ffn.w_up.weight")
        self.ffn_down = self.torch_to_tt("ffn.w_down.weight")

    def _modulate(self, x, scale, shift):
        ck = self.compute_kernel_config
        return ttnn.add(ttnn.multiply(_rms_norm(x, ck), ttnn.add(scale, 1.0)), shift)

    def __call__(self, x, c_l, cos, sin, attn_mask):
        ck = self.compute_kernel_config
        lin = lambda t, w: ttnn.linear(t, w, compute_kernel_config=ck, dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)
        mod = lin(ttnn.silu(c_l), self.adaln_w)  # [B,N,6d]
        sa, ca, ga, sf, cf, gf = ttnn.chunk(mod, 6, dim=-1)
        ttnn.deallocate(mod)

        attn_in = self._modulate(x, ca, sa)
        attn_out = self.attn(attn_in, cos, sin, attn_mask)
        x = ttnn.add(x, ttnn.multiply(ga, attn_out))
        ttnn.deallocate(attn_out)

        ffn_in = self._modulate(x, cf, sf)
        h1, h2 = ttnn.chunk(lin(ffn_in, self.ffn_up), 2, dim=-1)
        ttnn.deallocate(ffn_in)
        ffn_out = lin(ttnn.multiply(ttnn.silu(h1), h2), self.ffn_down)
        ttnn.deallocate(h1); ttnn.deallocate(h2)
        x = ttnn.add(x, ttnn.multiply(gf, ffn_out))
        ttnn.deallocate(ffn_out)
        return x


class SWAAtomTransformerModel(Module):
    def __init__(self, n_heads: int, n_blocks: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.blocks = [
            SWAAtomBlock(n_heads, self.scope(f"blocks.{i}"), compute_kernel_config)
            for i in range(n_blocks)
        ]

    def __call__(self, q_l, c_l, cos, sin, attn_mask):
        for block in self.blocks:
            q_l = block(q_l, c_l, cos, sin, attn_mask)
        return q_l


# Atom 3D-RoPE config (DiffusionModule defaults).
ATOM_ROPE = dict(n_spatial_per_axis=2, n_uid_pairs=10, spatial_base_freq=20.0, uid_base_freq=10000.0)


class SWAAtomTransformer(TorchWrapper):
    """Token-free SWA atom transformer (torch in/out).
    forward(q_l[B,N,d], c_l[B,N,d], ref_pos[B,N,3], ref_space_uid[B,N]) -> [B,N,d]."""

    def __init__(self, n_blocks: int = 3, n_heads: int = 4, swa_window_size: int = 128, d_atom: int = 128):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.half_window = swa_window_size // 2
        self.head_dim = d_atom // n_heads

    def _create_module(self, weights: WeightScope) -> SWAAtomTransformerModel:
        return SWAAtomTransformerModel(self.n_heads, self.n_blocks, weights, self.compute_kernel_config)

    def forward(self, q_l, c_l, ref_pos, ref_space_uid):
        n = q_l.shape[1]
        cos, sin = build_3d_rope_tables(ref_pos, ref_space_uid, self.head_dim, **ATOM_ROPE)
        mask = band_mask(n, self.half_window)
        out = self.module(
            self._from_torch(q_l), self._from_torch(c_l),
            self._from_torch(cos.float()), self._from_torch(sin.float()),
            self._from_torch(mask.float()),
        )
        return self._to_torch(out)


# ===========================================================================
# Atom encoder / decoder wrappers + DiffusionModule
# ===========================================================================

ATOM_FEATURE_DIM = 389  # 3 + 1 + 1 + 128 + 4*64
SIGMA_DATA = 16.0


class AtomEncoder(Module):
    """ref atom features (+ noisy coords) -> token-aggregated repr a, atom repr q, c."""

    def __init__(self, n_heads, n_blocks, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.atom_linear_w = self.torch_to_tt("atom_linear.weight")
        self.atom_norm_w = self.torch_to_tt("atom_norm.weight")
        self.atom_norm_b = self.torch_to_tt("atom_norm.bias")
        self.coords_w = self.torch_to_tt("coords_linear.weight")
        self.a2t_w = self.torch_to_tt("atom_to_token_linear.weight")
        self.transformer = SWAAtomTransformerModel(n_heads, n_blocks, self.scope("atom_transformer"), compute_kernel_config)

    def __call__(self, atom_feats, r_input, cos, sin, band, scatter_m):
        ck = self.compute_kernel_config
        lin = lambda t, w: ttnn.linear(t, w, compute_kernel_config=ck, dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)
        c = ttnn.layer_norm(lin(atom_feats, self.atom_linear_w), weight=self.atom_norm_w, bias=self.atom_norm_b, epsilon=1e-5, compute_kernel_config=ck)
        q = ttnn.add(c, lin(r_input, self.coords_w))
        q = self.transformer(q, c, cos, sin, band)
        q_to_a = ttnn.relu(lin(q, self.a2t_w))  # [B,N,d_token]
        a = ttnn.matmul(scatter_m, q_to_a, compute_kernel_config=ck, dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)
        ttnn.deallocate(q_to_a)
        return a, q, c


class AtomDecoder(Module):
    """token repr a + atom skip (q,c) -> per-atom coordinate update r_update [B,N,3]."""

    def __init__(self, n_heads, n_blocks, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.t2a_w = self.torch_to_tt("token_to_atom_linear.weight")
        self.norm_w = self.torch_to_tt("norm.weight")
        self.norm_b = self.torch_to_tt("norm.bias")
        self.out_w = self.torch_to_tt("output_linear.weight")
        self.transformer = SWAAtomTransformerModel(n_heads, n_blocks, self.scope("atom_transformer"), compute_kernel_config)

    def __call__(self, a, q_skip, c_skip, cos, sin, band, gather_g):
        ck = self.compute_kernel_config
        lin = lambda t, w: ttnn.linear(t, w, compute_kernel_config=ck, dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)
        a_to_q = lin(a, self.t2a_w)  # [B,L,d_atom]
        a_to_q = ttnn.matmul(gather_g, a_to_q, compute_kernel_config=ck, dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)  # [B,N,d_atom]
        q = ttnn.add(q_skip, a_to_q)
        q = self.transformer(q, c_skip, cos, sin, band)
        q = ttnn.layer_norm(q, weight=self.norm_w, bias=self.norm_b, epsilon=1e-5, compute_kernel_config=ck)
        return lin(q, self.out_w)  # [B,N,3]


class DiffusionModuleModel(Module):
    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        ck = compute_kernel_config
        self.conditioning = DiffusionConditioningModel(self.scope("conditioning"), ck)
        self.atom_encoder = AtomEncoder(4, 3, self.scope("atom_encoder"), ck)
        self.atom_decoder = AtomDecoder(4, 3, self.scope("atom_decoder"), ck)
        self.token_transformer = DiffusionTransformerModel(16, 12, self.scope("token_transformer"), ck)
        self.s_step_norm_w = self.torch_to_tt("s_step_norm.weight")
        self.s_step_norm_b = self.torch_to_tt("s_step_norm.bias")
        self.s_to_token_w = self.torch_to_tt("s_to_token.weight")
        self.token_norm_w = self.torch_to_tt("token_norm.weight")
        self.token_norm_b = self.torch_to_tt("token_norm.bias")

    def __call__(self, s_inputs, z_trunk, relpos, n_raw, atom_feats, r_input,
                 cos, sin, band, scatter_m, gather_g):
        ck = self.compute_kernel_config
        lin = lambda t, w: ttnn.linear(t, w, compute_kernel_config=ck, dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)
        s, z = self.conditioning(s_inputs, z_trunk, relpos, n_raw)
        a, q_skip, c_skip = self.atom_encoder(atom_feats, r_input, cos, sin, band, scatter_m)
        s_step = ttnn.layer_norm(s, weight=self.s_step_norm_w, bias=self.s_step_norm_b, epsilon=1e-5, compute_kernel_config=ck)
        a = ttnn.add(a, lin(s_step, self.s_to_token_w))
        a = self.token_transformer(a, s, z)
        a = ttnn.layer_norm(a, weight=self.token_norm_w, bias=self.token_norm_b, epsilon=1e-5, compute_kernel_config=ck)
        return self.atom_decoder(a, q_skip, c_skip, cos, sin, band, gather_g)


class DiffusionModule(TorchWrapper):
    """One diffusion denoising step. forward(x_noisy[B,N,3], t_hat, feats...) -> x_denoised[B,N,3]."""

    def __init__(self, sigma_data: float = SIGMA_DATA):
        super().__init__()
        self.sigma_data = sigma_data

    def _create_module(self, weights: WeightScope) -> DiffusionModuleModel:
        self._fw = weights["conditioning.fourier.w"]
        self._fb = weights["conditioning.fourier.b"]
        return DiffusionModuleModel(weights, self.compute_kernel_config)

    def forward(self, x_noisy, t_hat, ref_pos, ref_charge, ref_mask, ref_element,
                ref_atom_name_chars, ref_space_uid, tok_idx, s_inputs, z_trunk, relpos):
        sigma, B, N = self.sigma_data, x_noisy.shape[0], x_noisy.shape[1]
        L = s_inputs.shape[1]
        t = torch.as_tensor(t_hat, dtype=torch.float32).reshape(-1)
        if t.numel() == 1:
            t = t.expand(B)

        # host: noise embedding, normalized coords, atom features, rope/band/scatter/gather
        t_noise = 0.25 * torch.log((t / sigma).clamp(min=1e-20))
        n_raw = torch.cos(2.0 * math.pi * (t_noise[:, None] * self._fw[None, :] + self._fb[None, :])).float()
        denom = torch.sqrt(t * t + sigma * sigma)
        r_noisy = x_noisy / denom[:, None, None]
        r_input = torch.cat([r_noisy, torch.zeros_like(r_noisy)], dim=-1)  # [B,N,6]
        atom_feats = torch.cat([
            ref_pos, ref_charge.unsqueeze(-1), ref_mask.unsqueeze(-1).float(),
            ref_element, ref_atom_name_chars.reshape(B, N, -1),
        ], dim=-1)  # [B,N,389]
        cos, sin = build_3d_rope_tables(ref_pos, ref_space_uid, 32, **ATOM_ROPE)
        band = band_mask(N, 64)
        oh = F.one_hot(tok_idx[0].long(), L).float() * ref_mask[0].float()[:, None]  # [N,L]
        gather_g = oh.unsqueeze(0)  # [1,N,L]
        scatter_m = (oh / oh.sum(0).clamp(min=1)).t().unsqueeze(0)  # [1,L,N]

        ft = self._from_torch
        r_update = self.module(
            ft(s_inputs), ft(z_trunk), ft(relpos), ft(n_raw), ft(atom_feats.float()),
            ft(r_input.float()), ft(cos.float()), ft(sin.float()), ft(band.float()),
            ft(scatter_m.float()), ft(gather_g.float()),
        )
        r_update = self._to_torch(r_update)

        sigma2, t2 = sigma * sigma, (t * t)
        out = (sigma2 / (sigma2 + t2))[:, None, None] * x_noisy
        out = out + ((sigma * t) / torch.sqrt(sigma2 + t2))[:, None, None] * r_update
        return out


# ===========================================================================
# Distogram + Confidence heads
# ===========================================================================


class DistogramHead(Module):
    """Linear on the symmetrized pair: distogram_head(z + z.transpose(L axes))."""

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.w = self.torch_to_tt("weight")
        self.b = self.torch_to_tt("bias", transform=_ROW)

    def __call__(self, z: ttnn.Tensor) -> ttnn.Tensor:
        zs = ttnn.add(z, ttnn.permute(z, (0, 2, 1, 3)))
        return ttnn.linear(zs, self.w, bias=self.b, compute_kernel_config=self.compute_kernel_config,
                           dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)


class DistogramHeadModel(TorchWrapper):
    def _create_module(self, weights: WeightScope) -> DistogramHead:
        return DistogramHead(weights, self.compute_kernel_config)

    def forward(self, z):
        return self._to_torch(self.module(self._from_torch(z)))


class RowAttentionPooling(Module):
    """Per-row scalar attention over columns, weighted sum of z, out_proj."""

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.attn_w = self.torch_to_tt("attn_proj.weight")
        self.out_w = self.torch_to_tt("out_proj.weight")

    def __call__(self, z: ttnn.Tensor) -> ttnn.Tensor:
        ck = self.compute_kernel_config
        scores = ttnn.linear(z, self.attn_w, compute_kernel_config=ck, dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)  # [B,L,L,1]
        scores = ttnn.softmax(scores, dim=-2)  # over columns m (the L axis at dim 2)
        weights = ttnn.permute(scores, (0, 1, 3, 2))  # [B,L,1,L]
        ttnn.deallocate(scores)
        pooled = ttnn.matmul(weights, z, compute_kernel_config=ck, dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)  # [B,L,1,d]
        ttnn.deallocate(weights)
        pooled = ttnn.squeeze(pooled, 2)  # [B,L,d]
        return ttnn.linear(pooled, self.out_w, compute_kernel_config=ck, dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)


class ConfidenceHeadModel(Module):
    """Learned core: build pair from s/z, refine with folding trunk, emit logits."""

    def __init__(self, conf_trunk_layers: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        ck = compute_kernel_config
        self.s_in_norm_w = self.torch_to_tt("s_inputs_norm.weight")
        self.s_in_norm_b = self.torch_to_tt("s_inputs_norm.bias")
        self.s_to_z_w = self.torch_to_tt("s_to_z.weight")
        self.s_to_z_t_w = self.torch_to_tt("s_to_z_transpose.weight")
        self.prod_in1_w = self.torch_to_tt("s_to_z_prod_in1.weight")
        self.prod_in2_w = self.torch_to_tt("s_to_z_prod_in2.weight")
        self.prod_out_w = self.torch_to_tt("s_to_z_prod_out.weight")
        self.z_norm_w = self.torch_to_tt("z_norm.weight")
        self.z_norm_b = self.torch_to_tt("z_norm.bias")
        self.folding_trunk = FoldingTrunkModel(conf_trunk_layers, self.scope("folding_trunk"), ck)
        self.row_pool = RowAttentionPooling(self.scope("row_attention_pooling"), ck)
        self.pae_ln_w = self.torch_to_tt("pae_ln.weight"); self.pae_ln_b = self.torch_to_tt("pae_ln.bias")
        self.pae_w = self.torch_to_tt("pae_head.weight")
        self.pde_ln_w = self.torch_to_tt("pde_ln.weight"); self.pde_ln_b = self.torch_to_tt("pde_ln.bias")
        self.pde_w = self.torch_to_tt("pde_head.weight")
        self.plddt_ln_w = self.torch_to_tt("plddt_ln.weight"); self.plddt_ln_b = self.torch_to_tt("plddt_ln.bias")
        self.resolved_ln_w = self.torch_to_tt("resolved_ln.weight"); self.resolved_ln_b = self.torch_to_tt("resolved_ln.bias")

    def __call__(self, s_inputs, z, pair_dist_embed, gather_g, w_plddt, w_resolved):
        ck = self.compute_kernel_config
        lin = lambda t, w: ttnn.linear(t, w, compute_kernel_config=ck, dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)
        ln = lambda t, w, b: ttnn.layer_norm(t, weight=w, bias=b, epsilon=1e-5, compute_kernel_config=ck)
        L = z.shape[1]

        s_n = ln(s_inputs, self.s_in_norm_w, self.s_in_norm_b)
        zb = ln(z, self.z_norm_w, self.z_norm_b)
        zb = ttnn.add(zb, ttnn.unsqueeze(lin(s_n, self.s_to_z_w), 2))       # [B,L,1,256]
        zb = ttnn.add(zb, ttnn.unsqueeze(lin(s_n, self.s_to_z_t_w), 1))     # [B,1,L,256]
        # outer product: in1[:,:,None,:] * in2[:,None,:,:]  (expand both to [B,L,L,256])
        a = ttnn.repeat(ttnn.unsqueeze(lin(s_n, self.prod_in1_w), 2), (1, 1, L, 1))
        b = ttnn.repeat(ttnn.unsqueeze(lin(s_n, self.prod_in2_w), 1), (1, L, 1, 1))
        zb = ttnn.add(zb, lin(ttnn.multiply(a, b), self.prod_out_w))
        ttnn.deallocate(a); ttnn.deallocate(b)
        pair = ttnn.add(zb, pair_dist_embed)
        # folding_trunk deallocates its input; clone so we keep `pair` for the
        # residual (reference: pair = pair + folding_trunk(pair)).
        pair = ttnn.add(pair, self.folding_trunk(ttnn.clone(pair, memory_config=ttnn.DRAM_MEMORY_CONFIG)))

        single = self.row_pool(pair)
        pae_logits = lin(ln(pair, self.pae_ln_w, self.pae_ln_b), self.pae_w)
        pde_logits = lin(ln(pair, self.pde_ln_w, self.pde_ln_b), self.pde_w)

        # per-atom plddt/resolved: gather single->atoms, then per-atom weight matmul
        s_at = ttnn.matmul(gather_g, single, compute_kernel_config=ck, dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)  # [B,A,768]
        plddt_in = ttnn.unsqueeze(ln(s_at, self.plddt_ln_w, self.plddt_ln_b), 2)   # [B,A,1,768]
        resolved_in = ttnn.unsqueeze(ln(s_at, self.resolved_ln_w, self.resolved_ln_b), 2)
        plddt_logits = ttnn.squeeze(ttnn.matmul(plddt_in, w_plddt, compute_kernel_config=ck, dtype=ttnn.bfloat16), 2)
        resolved_logits = ttnn.squeeze(ttnn.matmul(resolved_in, w_resolved, compute_kernel_config=ck, dtype=ttnn.bfloat16), 2)
        return pae_logits, pde_logits, plddt_logits, resolved_logits


class ConfidenceHead(TorchWrapper):
    """Confidence head core (torch in/out). Returns pae/pde/plddt/resolved logits.

    Host-side: distogram-bin embedding (learned table indexed by binned predicted
    distances), per-atom weight gather (plddt/resolved weights by intra-token idx),
    and the token->atom gather matrix.
    """

    def __init__(self, conf_trunk_layers: int = 4, min_dist: float = 3.25,
                 max_dist: float = 50.75, distogram_bins: int = 39):
        super().__init__()
        self.conf_trunk_layers = conf_trunk_layers
        self.boundaries = torch.linspace(min_dist, max_dist, distogram_bins - 1)

    def _create_module(self, weights: WeightScope) -> ConfidenceHeadModel:
        self._dist_embed = weights["dist_bin_pairwise_embed.weight"]  # [bins, 256]
        self._plddt_weight = weights["plddt_weight"]                  # [23, 768, 50]
        self._resolved_weight = weights["resolved_weight"]            # [23, 768, 2]
        return ConfidenceHeadModel(self.conf_trunk_layers, weights, self.compute_kernel_config)

    def forward(self, s_inputs, z, rep_coords, atom_to_token, intra_idx):
        B, L = z.shape[0], z.shape[1]
        # distogram bins from predicted representative-atom distances
        d = torch.cdist(rep_coords, rep_coords, compute_mode="donot_use_mm_for_euclid_dist")
        bins = (d.unsqueeze(-1) > self.boundaries).sum(-1).long()  # [B,L,L]
        pair_dist_embed = self._dist_embed[bins]  # [B,L,L,256]

        A = atom_to_token.shape[1]
        oh = F.one_hot(atom_to_token[0].long(), L).float()  # [A,L]
        gather_g = oh.unsqueeze(0)  # [1,A,L]
        w_plddt = self._plddt_weight[intra_idx[0]].unsqueeze(0)        # [1,A,768,50]
        w_resolved = self._resolved_weight[intra_idx[0]].unsqueeze(0)  # [1,A,768,2]

        ft = self._from_torch
        pae, pde, plddt, resolved = self.module(
            ft(s_inputs), ft(z), ft(pair_dist_embed.float()), ft(gather_g),
            ft(w_plddt.float()), ft(w_resolved.float()),
        )
        return tuple(self._to_torch(t) for t in (pae, pde, plddt, resolved))
