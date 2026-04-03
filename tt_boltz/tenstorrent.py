import torch, ttnn, atexit
from torch import nn
from typing import Callable, Mapping
from math import pi
from functools import lru_cache
from types import MappingProxyType

TRIANGLE_MULT_CHUNK_SIZE = 32
TRIANGLE_ATT_CHUNK_SIZE_FAST = 1024
TRIANGLE_ATT_CHUNK_SIZE = 512
OPM_CHUNK_SIZE = 256
MSA_CHUNK_SIZE = 512
TRANSITION_W_CHUNK_SIZE = 1024
SEQ_LEN_MORE_CHUNKING = 1536
_FAST_MODE = False
TRIANGLE_MULT_L1_MAX_SEQ_FAST = 672
TRIANGLE_MULT_L1_MAX_SEQ = 352
SDPA_CHUNK_TILE = 32
SDPA_CHUNK_MAX = 256

PAIRFORMER_PAD_MULTIPLE = 64  # Pad token dim to this multiple to avoid kernel recompilation
MSA_PAD_MULTIPLE = 1024  # Pad MSA dim to this multiple to avoid kernel recompilation
MAX_ATOMS_PER_TOKEN = 14  # Upper bound on atoms per residue (Trp=14); ties atom bucket to seq_len bucket

ATOM_WINDOW = 32
ATOM_DIM = 128
ATOM_N_HEADS = 4
ATOM_N_LAYERS = 3
TOKEN_DIM = 2 * 384
TOKEN_N_HEADS = 16
TOKEN_N_LAYERS = 24

CORE_GRID_MAIN = ttnn.CoreGrid(y=10, x=11)
CORE_GRID_WIDE = ttnn.CoreGrid(y=10, x=11)
CORE_GRID_ATTN_BIAS = ttnn.CoreGrid(y=9, x=11)
CORE_GRID_ATTN_OUT = ttnn.CoreGrid(y=6, x=11)
CORE_GRID_REDUCED = ttnn.CoreGrid(y=8, x=11)
MAX_COMPUTE_GRID_X = 11
MAX_COMPUTE_GRID_Y = 10

def _dtype():
    return ttnn.bfloat8_b if _FAST_MODE else ttnn.bfloat16


def _adaln_memory_config(atom_level: bool, large_seq_len: bool) -> ttnn.MemoryConfig | None:
    if not atom_level:
        return None
    return ttnn.DRAM_MEMORY_CONFIG if large_seq_len else ttnn.L1_MEMORY_CONFIG


def _triangle_mul_memory_config(seq_len: int) -> ttnn.MemoryConfig:
    l1_max_seq = TRIANGLE_MULT_L1_MAX_SEQ_FAST if _FAST_MODE else TRIANGLE_MULT_L1_MAX_SEQ
    return ttnn.L1_MEMORY_CONFIG if seq_len <= l1_max_seq else ttnn.DRAM_MEMORY_CONFIG


@lru_cache(maxsize=1)
def _active_compute_grid_size() -> tuple[int, int]:
    return (MAX_COMPUTE_GRID_X, MAX_COMPUTE_GRID_Y)


@lru_cache(maxsize=None)
def _sdpa_program_config(q_chunk_size: int, k_chunk_size: int) -> ttnn.SDPAProgramConfig:
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=_active_compute_grid_size(),
        exp_approx_mode=False,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
    )


@lru_cache(maxsize=None)
def _capped_sdpa_chunk_size(seq_len: int) -> int:
    if seq_len <= 0:
        return SDPA_CHUNK_TILE
    return min(SDPA_CHUNK_MAX, ((seq_len + SDPA_CHUNK_TILE - 1) // SDPA_CHUNK_TILE) * SDPA_CHUNK_TILE)


@lru_cache(maxsize=None)
def _sdpa_program_config_for_lengths(q_len: int, k_len: int) -> ttnn.SDPAProgramConfig:
    return _sdpa_program_config(
        q_chunk_size=_capped_sdpa_chunk_size(q_len),
        k_chunk_size=_capped_sdpa_chunk_size(k_len),
    )


@lru_cache(maxsize=None)
def _triangle_mul_program_config(seq_len_tiles: int) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    gx, gy = _active_compute_grid_size()
    per_core_M = -(-seq_len_tiles // gy)
    per_core_N = -(-seq_len_tiles // gx)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=per_core_M,
        out_block_w=per_core_N,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def set_fast_mode(enabled: bool) -> None:
    """Set fast block-fp8 mode for the current worker process."""
    global _FAST_MODE
    _FAST_MODE = bool(enabled)

_device = None

def get_device():
    """Open (or return cached) TT device 0.

    For multi-device setups each worker process sets TT_VISIBLE_DEVICES
    before any ttnn call so that its single physical card appears as
    logical device 0.  All workers share the default on-disk kernel cache.
    """
    global _device
    if _device is None:
        _device = ttnn.open_device(device_id=0)
        _device.enable_program_cache()
    return _device


def cleanup():
    global _device
    if _device is not None:
        try:
            # Drain queued work before closing so teardown is deterministic.
            ttnn.synchronize_device(_device)
        except Exception:
            pass
        ttnn.close_device(_device)
        _device = None


atexit.register(cleanup)


class WeightScope:
    """Immutable scoped view over a flat checkpoint state-dict."""

    def __init__(self, data: Mapping[str, torch.Tensor]):
        self._data = MappingProxyType(dict(data))

    @classmethod
    def wrap(cls, data: Mapping[str, torch.Tensor] | "WeightScope") -> "WeightScope":
        return data if isinstance(data, cls) else cls(data)

    @property
    def data(self) -> Mapping[str, torch.Tensor]:
        return self._data

    def as_dict(self) -> dict[str, torch.Tensor]:
        return dict(self._data)

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._data[key]

    def child(self, scope: str, strip_prefix: str = "") -> "WeightScope":
        if not scope:
            return self
        scope_prefix = f"{scope}."
        out = {}
        for key, value in self._data.items():
            if not key.startswith(scope_prefix):
                continue
            child_key = key[len(scope_prefix) :]
            if strip_prefix and child_key.startswith(strip_prefix):
                child_key = child_key[len(strip_prefix) :]
            out[child_key] = value
        return WeightScope(out)


Weights = Mapping[str, torch.Tensor] | WeightScope

class Module:
    def __init__(
        self,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        self.weights = WeightScope.wrap(state_dict)
        self.compute_kernel_config = compute_kernel_config
        self.device = get_device()

    def scope(self, scope: str, strip_prefix: str = "") -> WeightScope:
        return self.weights.child(scope, strip_prefix)

    def torch_to_tt(
        self,
        key: str,
        transform: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.t(),
        dtype=ttnn.bfloat16,
    ) -> ttnn.Tensor:
        return ttnn.from_torch(
            transform(self.weights[key]),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            dtype=dtype,
        )


class TriangleMultiplication(Module):
    def __init__(
        self,
        ending: bool,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.ending = ending
        self.in_norm_weight = self.torch_to_tt("norm_in.weight")
        self.in_norm_bias = self.torch_to_tt("norm_in.bias")
        self.out_norm_weight = self.torch_to_tt("norm_out.weight")
        self.out_norm_bias = self.torch_to_tt("norm_out.bias")
        g_in_t, p_in_t = [
            self.weights[k].t() for k in ["g_in.weight", "p_in.weight"]
        ]
        C = TRIANGLE_MULT_CHUNK_SIZE
        self.n_pairs = g_in_t.shape[1] // C // 2
        self.gp_in_weight_fused_chunks = [
            ttnn.from_torch(
                torch.cat(
                    [
                        g_in_t[:, i * C : (i + 1) * C],
                        g_in_t[:, (i + self.n_pairs) * C : (i + self.n_pairs + 1) * C],
                        p_in_t[:, i * C : (i + 1) * C],
                        p_in_t[:, (i + self.n_pairs) * C : (i + self.n_pairs + 1) * C],
                    ],
                    dim=1,
                ),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                dtype=ttnn.bfloat16,
            )
            for i in range(self.n_pairs)
        ]
        self.g_out_weight = self.torch_to_tt("g_out.weight")
        self.out_p_weight = self.torch_to_tt("p_out.weight")

    def _transform_chunk(
        self, chunk: ttnn.Tensor, permute_dims: tuple[int, ...], memory_config: ttnn.MemoryConfig
    ) -> ttnn.Tensor:
        old = chunk
        for op, *args in (
            [
                (ttnn.typecast, ttnn.bfloat16),
                (ttnn.permute, permute_dims),
                (ttnn.typecast, ttnn.bfloat8_b),
                (ttnn.reallocate,),
            ] if _FAST_MODE else [
                (ttnn.permute, permute_dims),
                (ttnn.reallocate,),
            ]
        ):
            chunk = op(chunk, *args, memory_config=memory_config)
            ttnn.deallocate(old)
            old = chunk
        return chunk

    def __call__(self, x: ttnn.Tensor, mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        x_norm_in = ttnn.layer_norm(
            x,
            weight=self.in_norm_weight,
            bias=self.in_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        H = x_norm_in.shape[1]
        memory_config = _triangle_mul_memory_config(H)
        seq_len_tiles = (H + 31) // 32
        program_config = _triangle_mul_program_config(seq_len_tiles)
        if H > SEQ_LEN_MORE_CHUNKING:
            # Compact large input activation for better large-sequence placement.
            x_norm_in = ttnn.reallocate(x_norm_in)
        # Unsqueeze mask once before chunk loop (mask is [1,S,S] or [1,S])
        mask_u = ttnn.unsqueeze(mask, -1) if mask is not None else None
        for i in range(self.n_pairs):
            gp_in_fused = ttnn.experimental.minimal_matmul(
                x_norm_in,
                self.gp_in_weight_fused_chunks[i],
                memory_config=memory_config,
                dtype=_dtype(),
                compute_kernel_config=self.compute_kernel_config,
            )
            g_in_a, g_in_b, p_in_a, p_in_b = ttnn.chunk(gp_in_fused, chunks=4, dim=-1)
            ttnn.deallocate(gp_in_fused)
            a_chunk = ttnn.multiply_(
                p_in_a, g_in_a, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]
            )
            b_chunk = ttnn.multiply_(
                p_in_b, g_in_b, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]
            )
            ttnn.deallocate(g_in_a)
            ttnn.deallocate(g_in_b)
            if mask_u is not None:
                a_chunk = ttnn.multiply_(a_chunk, mask_u)

            a_chunk = self._transform_chunk(
                a_chunk, (0, 3) + ((2, 1) if self.ending else (1, 2)), memory_config=memory_config,
            )
            b_chunk = self._transform_chunk(
                b_chunk, (0, 3) + ((1, 2) if self.ending else (2, 1)), memory_config=memory_config,
            )
            x_chunk = ttnn.matmul(
                a_chunk,
                b_chunk,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=memory_config,
                program_config=program_config,
                dtype=ttnn.bfloat16,
            )
            ttnn.deallocate(a_chunk)
            ttnn.deallocate(b_chunk)
            x_chunk = ttnn.permute(x_chunk, (0, 2, 3, 1), memory_config=memory_config)
            if i == 0:
                x = ttnn.clone(x_chunk, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            else:
                x_old = x
                x = ttnn.concat([x_old, x_chunk], dim=-1)
                ttnn.deallocate(x_old)
            ttnn.deallocate(x_chunk)
        x = ttnn.layer_norm(
            x,
            weight=self.out_norm_weight,
            bias=self.out_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        if H > SEQ_LEN_MORE_CHUNKING:
            # Reduce DRAM fragmentation before the two largest output projections.
            x = ttnn.reallocate(x)
            x_norm_in = ttnn.reallocate(x_norm_in)
        p_out = ttnn.linear(
            x,
            self.out_p_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=_dtype(),
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_WIDE,
        )
        ttnn.deallocate(x)
        g_out = ttnn.linear(
            x_norm_in,
            self.g_out_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=_dtype(),
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_WIDE,
        )
        ttnn.deallocate(x_norm_in)
        x = ttnn.multiply_(
            p_out, g_out, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]
        )
        return x


class TriangleAttention(Module):
    def __init__(
        self,
        head_dim: int,
        n_heads: int,
        ending: bool,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
        affinity: bool = False,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.ending = ending
        self.affinity = affinity
        self.scale = self.head_dim**0.5
        self.layer_norm_weight = self.torch_to_tt("layer_norm.weight")
        self.layer_norm_bias = self.torch_to_tt("layer_norm.bias")
        self.o_weight = self.torch_to_tt("linear_o.weight")
        self.bias_weight = ttnn.multiply_(self.torch_to_tt("linear.weight"), self.scale)
        self.qkv_weight = ttnn.from_torch(
            torch.cat(
                [
                    self.weights["linear_q.weight"],
                    self.weights["linear_k.weight"],
                    self.weights["linear_v.weight"],
                ],
                dim=0,
            ).t(),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            dtype=_dtype(),
        )
        self.g_weight = self.torch_to_tt("linear_g.weight", dtype=_dtype())

    def __call__(self, x: ttnn.Tensor, attn_mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        x = ttnn.reshape(x, tuple(x.shape)[1:])
        if self.ending:
            x = ttnn.permute(x, (1, 0, 2))  # THIS CAUSES CACHE -> RESHAPE PROBLEM
        x = ttnn.layer_norm(
            x,
            weight=self.layer_norm_weight,
            bias=self.layer_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        triangle_bias = ttnn.linear(
            x,
            self.bias_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            core_grid=CORE_GRID_ATTN_BIAS,
        )
        triangle_bias = ttnn.unsqueeze(triangle_bias, 0)
        triangle_bias = ttnn.permute(triangle_bias, (0, 3, 1, 2))

        def attend(qkv_in, bias):
            qkv_in = ttnn.unsqueeze(qkv_in, 1)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv_in, num_heads=self.n_heads, num_kv_heads=self.n_heads,
                transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(qkv_in)
            o = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, is_causal=False, scale=self.scale**-1,
                program_config=_sdpa_program_config_for_lengths(q.shape[2], k.shape[2]),
            )
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            o_heads = ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(o)
            return ttnn.squeeze(o_heads, 1)

        def gate_and_project(o_in: ttnn.Tensor, g_in: ttnn.Tensor) -> ttnn.Tensor:
            o_in = ttnn.multiply_(o_in, g_in, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            ttnn.deallocate(g_in)
            x_out = ttnn.linear(
                o_in,
                self.o_weight,
                compute_kernel_config=self.compute_kernel_config,
                dtype=_dtype(),
                core_grid=CORE_GRID_ATTN_OUT,
            )
            ttnn.deallocate(o_in)
            return x_out

        S = x.shape[0]
        need_chunk = S > SEQ_LEN_MORE_CHUNKING and (self.affinity or not _FAST_MODE)
        if need_chunk:
            if not self.affinity and attn_mask is not None:
                triangle_bias = ttnn.add(triangle_bias, attn_mask)
            chunk = TRIANGLE_ATT_CHUNK_SIZE_FAST if _FAST_MODE else TRIANGLE_ATT_CHUNK_SIZE
            parts = []
            for s in range(0, S, chunk):
                end = min(s + chunk, S)
                x_chunk = x[s:end, :, :]
                qkv_chunk = ttnn.experimental.minimal_matmul(
                    input_tensor=x_chunk,
                    weight_tensor=self.qkv_weight,
                    compute_kernel_config=self.compute_kernel_config,
                    dtype=_dtype(),
                )
                g_chunk = ttnn.experimental.minimal_matmul(
                    input_tensor=x_chunk,
                    weight_tensor=self.g_weight,
                    compute_kernel_config=self.compute_kernel_config,
                    dtype=_dtype(),
                )
                if self.affinity:
                    bias = ttnn.add(triangle_bias, attn_mask[s:end, :, :])
                    o_chunk = attend(qkv_chunk, bias)
                    ttnn.deallocate(bias)
                else:
                    o_chunk = attend(qkv_chunk, triangle_bias)
                ttnn.deallocate(qkv_chunk)
                parts.append(gate_and_project(o_chunk, g_chunk))
            ttnn.deallocate(x)
            ttnn.deallocate(triangle_bias)
            x = ttnn.concat(parts, dim=0)
            del parts
        else:
            qkv = ttnn.experimental.minimal_matmul(
                input_tensor=x,
                weight_tensor=self.qkv_weight,
                compute_kernel_config=self.compute_kernel_config,
                dtype=_dtype(),
            )
            g = ttnn.experimental.minimal_matmul(
                input_tensor=x,
                weight_tensor=self.g_weight,
                compute_kernel_config=self.compute_kernel_config,
                dtype=_dtype(),
            )
            ttnn.deallocate(x)
            if attn_mask is not None:
                triangle_bias = ttnn.add(triangle_bias, attn_mask)
            o = attend(qkv, triangle_bias)
            ttnn.deallocate(qkv)
            ttnn.deallocate(triangle_bias)
            x = gate_and_project(o, g)
        if self.ending:
            x = ttnn.permute(x, (1, 0, 2))
        x = ttnn.reshape(x, (1, *x.shape))
        return x


class AttentionPairBias(Module):
    def __init__(
        self,
        head_dim: int,
        n_heads: int,
        compute_pair_bias: bool,
        atom_level: bool,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.compute_pair_bias = compute_pair_bias
        self.atom_level = atom_level
        if atom_level:
            self.q_weight = self.torch_to_tt("proj_q.weight", dtype=_dtype())
            self.q_bias = self.torch_to_tt("proj_q.bias", dtype=_dtype())
            kv_weight = torch.cat([self.weights["proj_k.weight"], self.weights["proj_v.weight"]], dim=0)
            self.kv_weight = ttnn.from_torch(
                kv_weight.t(),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                dtype=_dtype(),
            )
        else:
            qkv_weight = torch.cat(
                [self.weights["proj_q.weight"], self.weights["proj_k.weight"], self.weights["proj_v.weight"]],
                dim=0,
            )
            head_dim_padding = -head_dim % 32
            padded_head_dim = head_dim + head_dim_padding
            qkv_weight = qkv_weight.reshape(3 * self.n_heads, head_dim, -1)
            qkv_weight = torch.nn.functional.pad(qkv_weight, (0, 0, 0, head_dim_padding), mode='constant', value=0)
            qkv_weight = qkv_weight.reshape(3 * self.n_heads * padded_head_dim, -1)
            self.qkv_weight = ttnn.from_torch(
                qkv_weight.t(),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                dtype=ttnn.bfloat16,
            )
            q_bias = self.weights["proj_q.bias"]
            q_bias = q_bias.reshape(self.n_heads, head_dim)
            q_bias = torch.nn.functional.pad(q_bias, (0, head_dim_padding), mode='constant', value=0)
            q_bias = q_bias.reshape(self.n_heads * padded_head_dim)
            qkv_bias = torch.cat([q_bias, torch.zeros(2 * self.n_heads * padded_head_dim, dtype=q_bias.dtype, device=q_bias.device)])
            self.qkv_bias = ttnn.from_torch(
                qkv_bias,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                dtype=ttnn.bfloat16,
            )
        self.g_weight = self.torch_to_tt("proj_g.weight")
        if compute_pair_bias:
            self.z_norm_weight = self.torch_to_tt("proj_z.0.weight")
            self.z_norm_bias = self.torch_to_tt("proj_z.0.bias")
            self.z_weight = ttnn.multiply_(
                self.torch_to_tt("proj_z.1.weight"), self.head_dim**0.5
            )
        self.o_weight = self.torch_to_tt("proj_o.weight")

    def __call__(
        self,
        s: ttnn.Tensor,
        z: ttnn.Tensor,
        keys_indexing: ttnn.Tensor | None = None,
        seq_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        if not self.atom_level:
            qkv = ttnn.linear(
                s,
                self.qkv_weight,
                bias=self.qkv_bias,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
            )
            qkv = ttnn.unsqueeze(qkv, 1)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv,
                num_heads=self.n_heads,
                num_kv_heads=self.n_heads,
                transpose_k_heads=False,
            )
            ttnn.deallocate(qkv)
            if self.compute_pair_bias:
                z = ttnn.layer_norm(
                    z,
                    weight=self.z_norm_weight,
                    bias=self.z_norm_bias,
                    epsilon=1e-5,
                    compute_kernel_config=self.compute_kernel_config,
                )
                z = ttnn.linear(
                    z,
                    self.z_weight,
                    compute_kernel_config=self.compute_kernel_config,
                    core_grid=CORE_GRID_REDUCED,
                )
                z = ttnn.permute(z, (0, 3, 1, 2))
            if seq_mask is not None:
                z = ttnn.add_(z, seq_mask)
            o = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=z,
                is_causal=False,
                scale=self.head_dim**-0.5,
                program_config=_sdpa_program_config_for_lengths(q.shape[2], k.shape[2]),
            )
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            o = o[:, :, :, :self.head_dim]
            o = ttnn.permute(o, (0, 1, 3, 2))
            o = ttnn.reshape(o, (o.shape[0], -1, o.shape[3]))
            o = ttnn.permute(o, (0, 2, 1))
        else:
            s = ttnn.to_memory_config(s, ttnn.DRAM_MEMORY_CONFIG, dtype=_dtype())
            B, K, W, D_S = s.shape
            s_kv = ttnn.reshape(s, (B, 2 * K, W // 2, -1))
            s_kv = ttnn.permute(s_kv, (0, 2, 3, 1))
            s_kv = ttnn.matmul(
                s_kv,
                keys_indexing,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
            )
            s_kv = ttnn.permute(s_kv, (0, 3, 1, 2))
            s_kv = ttnn.reshape(s_kv, (B, K, -1, D_S))

            q = ttnn.linear(
                s,
                self.q_weight,
                bias=self.q_bias,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
                dtype=_dtype(),
            )
            kv = ttnn.linear(
                s_kv,
                self.kv_weight,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
                dtype=_dtype(),
            )

            q = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)
            q = ttnn.pad(q, [[0, 0], [0, 0], [0, ATOM_DIM - ATOM_WINDOW], [0, 0]], 0.0)
            q = ttnn.to_layout(q, ttnn.TILE_LAYOUT, dtype=_dtype())
            q = ttnn.reshape(q, (B * K, 1, ATOM_DIM, -1))
            kv = ttnn.reshape(kv, (B * K, 1, ATOM_DIM, -1))
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(q, kv, num_heads=self.n_heads, num_kv_heads=self.n_heads, transpose_k_heads=False)
            _, H, S, D_Q = q.shape
            q = ttnn.reshape(q, (B, K * H, S, D_Q))
            k = ttnn.reshape(k, (B, K * H, S, D_Q))
            v = ttnn.reshape(v, (B, K * H, S, D_Q))
            q = q[:, :, :ATOM_WINDOW, :]
            z = ttnn.reshape(z, (1, -1, z.shape[2], z.shape[3]))
            o = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=z, is_causal=False, scale=self.head_dim**-0.5,
                program_config=_sdpa_program_config_for_lengths(q.shape[2], k.shape[2]),
            )
            o = ttnn.reshape(o, (B * K, H, W, D_Q))
            o = ttnn.experimental.nlp_concat_heads(o)
            o = ttnn.squeeze(o, 1)
            o = ttnn.reshape(o, (B, K, W, D_S))
        g = ttnn.linear(
            s,
            self.g_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        if _FAST_MODE:
            o = ttnn.typecast(o, ttnn.bfloat16)
        o = ttnn.multiply(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID], dtype=_dtype())
        ttnn.deallocate(g)
        x = ttnn.linear(
            o, self.o_weight, compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        ttnn.deallocate(o)
        return x


class Transition(Module):
    def __init__(
        self,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.norm_weight = self.torch_to_tt("norm.weight")
        self.norm_bias = self.torch_to_tt("norm.bias")
        self.fc1_weight = self.torch_to_tt("fc1.weight")
        self.fc2_weight = self.torch_to_tt("fc2.weight")
        self.fc3_weight = self.torch_to_tt("fc3.weight")

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        def swiglu(x):
            x_norm = ttnn.layer_norm(
                x,
                weight=self.norm_weight,
                bias=self.norm_bias,
                epsilon=1e-5,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            x_1 = ttnn.linear(
                x_norm,
                self.fc1_weight,
                activation="silu",
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=_dtype(),
                core_grid=CORE_GRID_MAIN,
            )
            x_2 = ttnn.linear(
                x_norm,
                self.fc2_weight,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=_dtype(),
                core_grid=CORE_GRID_MAIN,
            )
            ttnn.deallocate(x_norm)
            x = ttnn.multiply_(x_1, x_2)
            ttnn.deallocate(x_2)
            x_dram = ttnn.linear(
                x,
                self.fc3_weight,
                compute_kernel_config=self.compute_kernel_config,
                dtype=_dtype(),
                core_grid=CORE_GRID_REDUCED,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(x)
            return x_dram
        if len(x.shape) < 4:
            if x.shape[1] > SEQ_LEN_MORE_CHUNKING:
                return ttnn.concat([swiglu(x[b:b+1, :, :]) for b in range(x.shape[0])], dim=0)
            return swiglu(x)

        H, W = x.shape[1], x.shape[2]
        if W * x.shape[3] <= TOKEN_DIM * ATOM_DIM:
            chunk_h = 64 if _FAST_MODE else 32
        else:
            chunk_h = 32 if _FAST_MODE else 16
        chunks = ttnn.chunk(x, -(-H // chunk_h), dim=1)
        if W <= SEQ_LEN_MORE_CHUNKING:
            return ttnn.concat([swiglu(c) for c in chunks], dim=1)
        return ttnn.concat([
            ttnn.concat([swiglu(c[:, :, w:min(w+TRANSITION_W_CHUNK_SIZE, W), :]) for w in range(0, W, TRANSITION_W_CHUNK_SIZE)], dim=2)
            for c in chunks
        ], dim=1)


class PairformerLayer(Module):
    def __init__(
        self,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        att_head_dim: int | None,
        att_n_heads: int | None,
        transform_s: bool,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
        affinity: bool = False,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.transform_s = transform_s
        self.triangle_multiplication_start = TriangleMultiplication(
            False, self.scope("tri_mul_out"), compute_kernel_config
        )
        self.triangle_multiplication_end = TriangleMultiplication(
            True, self.scope("tri_mul_in"), compute_kernel_config
        )
        self.triangle_attention_start = TriangleAttention(
            tri_att_head_dim,
            tri_att_n_heads,
            False,
            self.scope("tri_att_start", "mha."),
            compute_kernel_config,
            affinity=affinity,
        )
        self.triangle_attention_end = TriangleAttention(
            tri_att_head_dim,
            tri_att_n_heads,
            True,
            self.scope("tri_att_end", "mha."),
            compute_kernel_config,
            affinity=affinity,
        )
        self.transition_z = Transition(
            self.scope("transition_z"), compute_kernel_config
        )
        if transform_s:
            self.pre_norm_s_weight = self.torch_to_tt("pre_norm_s.weight")
            self.pre_norm_s_bias = self.torch_to_tt("pre_norm_s.bias")
            self.attention_pair_bias = AttentionPairBias(
                att_head_dim,
                att_n_heads,
                True,
                False,
                self.scope("attention"),
                compute_kernel_config,
            )
            self.transition_s = Transition(
                self.scope("transition_s"), compute_kernel_config
            )

    def __call__(
        self, s: ttnn.Tensor | None, z: ttnn.Tensor, mask: ttnn.Tensor | None = None,
        attn_mask_start: ttnn.Tensor | None = None, attn_mask_end: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor | None, ttnn.Tensor]:
        z_update = self.triangle_multiplication_start(z, mask)
        z = ttnn.add_(z, z_update)
        ttnn.deallocate(z_update)

        z_update = self.triangle_multiplication_end(z, mask)
        z = ttnn.add_(z, z_update)
        ttnn.deallocate(z_update)

        z_update = self.triangle_attention_start(z, attn_mask_start)
        z = ttnn.add_(z, z_update)
        ttnn.deallocate(z_update)

        z_update = self.triangle_attention_end(z, attn_mask_end)
        z = ttnn.add_(z, z_update)
        ttnn.deallocate(z_update)

        z_update = self.transition_z(z)
        z = ttnn.add_(z, z_update)
        ttnn.deallocate(z_update)
        if self.transform_s:
            s_norm = ttnn.layer_norm(
                s,
                weight=self.pre_norm_s_weight,
                bias=self.pre_norm_s_bias,
                epsilon=1e-5,
                compute_kernel_config=self.compute_kernel_config,
            )
            s_update = self.attention_pair_bias(
                s_norm,
                z,
                seq_mask=attn_mask_start,  # same as end for non-affinity
            )
            ttnn.deallocate(s_norm)
            s = ttnn.add_(s, s_update)
            ttnn.deallocate(s_update)

            s_update = self.transition_s(s)
            s = ttnn.add_(s, s_update)
            ttnn.deallocate(s_update)
        return s, z


class Pairformer(Module):
    def __init__(
        self,
        n_blocks: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        att_head_dim: int | None,
        att_n_heads: int | None,
        transform_s: bool,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
        affinity: bool = False,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.blocks = [
            PairformerLayer(
                tri_att_head_dim,
                tri_att_n_heads,
                att_head_dim,
                att_n_heads,
                transform_s,
                self.scope(f"layers.{i}"),
                compute_kernel_config,
                affinity=affinity,
            )
            for i in range(n_blocks)
        ]

    def __call__(
        self, s: ttnn.Tensor | None, z: ttnn.Tensor, mask: ttnn.Tensor | None = None,
        attn_mask_start: ttnn.Tensor | None = None, attn_mask_end: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor | None, ttnn.Tensor]:
        for block in self.blocks:
            s, z = block(s, z, mask, attn_mask_start, attn_mask_end)
        return s, z


class AdaLN(Module):
    def __init__(
        self,
        atom_level: bool,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.atom_level = atom_level
        self.s_norm_weight = self.torch_to_tt("s_norm.weight")
        self.s_scale_weight = self.torch_to_tt("s_scale.weight")
        self.s_scale_bias = self.torch_to_tt("s_scale.bias")
        self.s_bias_weight = self.torch_to_tt("s_bias.weight")

    def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor, large_seq_len: bool = False) -> ttnn.Tensor:
        memory_config = _adaln_memory_config(self.atom_level, large_seq_len)
        if self.atom_level:
            a = ttnn.to_memory_config(a, memory_config=memory_config)
            s = ttnn.to_memory_config(s, memory_config=memory_config)
        a = ttnn.layer_norm(
            a, epsilon=1e-5, compute_kernel_config=self.compute_kernel_config
        )
        s = ttnn.layer_norm(
            s,
            weight=self.s_norm_weight,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        s_scale = ttnn.linear(
            s,
            self.s_scale_weight,
            bias=self.s_scale_bias,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
            #core_grid=ttnn.CoreGrid(y=10, x=11), CAUSES ACCURACY ISSUE
        )
        s_bias = ttnn.linear(
            s,
            self.s_bias_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
            #core_grid=ttnn.CoreGrid(y=10, x=11), CAUSES ACCURACY ISSUE
        )
        a = ttnn.multiply_(a, s_scale, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(s_scale)
        a = ttnn.add_(a, s_bias)
        ttnn.deallocate(s_bias)
        a = ttnn.to_memory_config(a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return a


class ConditionedTransitionBlock(Module):
    def __init__(
        self,
        atom_level: bool,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.atom_level = atom_level
        self.adaln = AdaLN(
            atom_level, self.scope("adaln"), compute_kernel_config
        )
        swish_chunk, gates_chunk = torch.chunk(self.weights["swish_gate.0.weight"], chunks=2, dim=0)
        self.swish_weight, self.gates_weight = [
            ttnn.from_torch(chunk.t(), layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
            for chunk in [swish_chunk, gates_chunk]
        ]
        self.a_to_b_weight = self.torch_to_tt("a_to_b.weight")
        self.b_to_a_weight = self.torch_to_tt("b_to_a.weight")
        self.output_projection_weight = self.torch_to_tt("output_projection.0.weight")
        self.output_projection_bias = self.torch_to_tt("output_projection.0.bias")

    def __call__(
        self, a: ttnn.Tensor, s: ttnn.Tensor, large_seq_len: bool = False
    ) -> ttnn.Tensor:
        a = self.adaln(a, s, large_seq_len=large_seq_len)
        a_swish = ttnn.linear(
            a,
            self.swish_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        gates = ttnn.linear(
            a,
            self.gates_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        a_swish = ttnn.multiply_(gates, a_swish, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        a_b = ttnn.linear(
            a,
            self.a_to_b_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        ttnn.deallocate(a)
        b = ttnn.multiply_(a_swish, a_b)
        ttnn.deallocate(a_b)
        s = ttnn.linear(
            s,
            self.output_projection_weight,
            bias=self.output_projection_bias,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        b_a = ttnn.linear(
            b,
            self.b_to_a_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        ttnn.deallocate(b)
        a = ttnn.multiply_(s, b_a, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(b_a)
        return a


class DiffusionTransformerLayer(Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        atom_level: bool,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.atom_level = atom_level
        self.s_o = None
        self.adaln = AdaLN(
            atom_level, self.scope("adaln"), compute_kernel_config
        )
        self.attn_pair_bias = AttentionPairBias(
            head_dim=dim // n_heads,
            n_heads=n_heads,
            compute_pair_bias=False,
            atom_level=atom_level,
            state_dict=self.scope("pair_bias_attn"),
            compute_kernel_config=compute_kernel_config,
        )
        self.output_projection_weight = self.torch_to_tt(
            "output_projection_linear.weight"
        )
        self.output_projection_bias = self.torch_to_tt("output_projection_linear.bias")
        self.transition = ConditionedTransitionBlock(
            atom_level,
            self.scope("transition"),
            compute_kernel_config,
        )

    def __call__(
        self,
        a: ttnn.Tensor,
        s: ttnn.Tensor,
        z: ttnn.Tensor,
        keys_indexing: ttnn.Tensor | None = None,
        large_seq_len: bool = False,
    ) -> ttnn.Tensor:
        b = self.adaln(a, s, large_seq_len=large_seq_len)
        if not self.atom_level:
            b = self.attn_pair_bias(b, z)
        else:
            b = self.attn_pair_bias(b, z, keys_indexing)
        if self.s_o is None:
            s_o = ttnn.linear(
                s,
                self.output_projection_weight,
                bias=self.output_projection_bias,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
                activation="sigmoid",
            )
            if self.atom_level:
                self.s_o = s_o
        else:
            s_o = self.s_o
        b = ttnn.multiply(s_o, b)
        a = ttnn.add(a, b)
        a_t = self.transition(a, s, large_seq_len=large_seq_len)
        a = ttnn.add(a, a_t)
        return a


class DiffusionTransformer(Module):
    def __init__(
        self,
        n_layers: int,
        dim: int,
        n_heads: int,
        atom_level: bool,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.layers = [
            DiffusionTransformerLayer(
                dim,
                n_heads,
                atom_level,
                self.scope(f"layers.{i}"),
                compute_kernel_config,
            )
            for i in range(n_layers)
        ]

    def __call__(
        self,
        a: ttnn.Tensor,
        s: ttnn.Tensor,
        z: ttnn.Tensor,
        keys_indexing: ttnn.Tensor | None = None,
        large_seq_len: bool = False,
    ) -> ttnn.Tensor:
        dim = z.shape[1] // len(self.layers)
        for i, layer in enumerate(self.layers):
            a = layer(
                a,
                s,
                z[:, i * dim : (i + 1) * dim, :, :],
                keys_indexing,
                large_seq_len=large_seq_len,
            )
        return a


class PairWeightedAveraging(Module):
    def __init__(
        self,
        head_dim: int,
        n_heads: int,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.m_norm_weight = self.torch_to_tt("norm_m.weight")
        self.m_norm_bias = self.torch_to_tt("norm_m.bias")
        self.z_norm_weight = self.torch_to_tt("norm_z.weight")
        self.z_norm_bias = self.torch_to_tt("norm_z.bias")
        self.m_weight = self.torch_to_tt("proj_m.weight")
        self.g_weight = self.torch_to_tt("proj_g.weight")
        self.z_weight = self.torch_to_tt("proj_z.weight")
        self.o_weight = self.torch_to_tt("proj_o.weight")

    def __call__(self, m: ttnn.Tensor, z: ttnn.Tensor, attn_mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        m = ttnn.reshape(m, tuple(m.shape)[1:])
        z = ttnn.reshape(z, tuple(z.shape)[1:])
        m = ttnn.layer_norm(
            m,
            weight=self.m_norm_weight,
            bias=self.m_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        z = ttnn.layer_norm(
            z,
            weight=self.z_norm_weight,
            bias=self.z_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        o_out = None
        for i in range(self.n_heads):
            b = ttnn.linear(
                z,
                self.z_weight[:, i : i + 1],
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
            )
            b = ttnn.permute(b, (2, 0, 1))
            if attn_mask is not None:
                b = ttnn.add_(b, ttnn.reshape(attn_mask, (1, 1, attn_mask.shape[-1])))
            w = ttnn.softmax(
                b,
                dim=-1,
                compute_kernel_config=self.compute_kernel_config,
                numeric_stable=True,
            )
            v = ttnn.linear(
                m,
                self.m_weight[:, i * self.head_dim : (i + 1) * self.head_dim],
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
            )
            # TODO: Inline with transpose_a=True after newest tt-metal release.
            v = ttnn.permute(v, (0, 2, 1))
            o = ttnn.matmul(
                v,
                w,
                transpose_b=True,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
            )
            ttnn.deallocate(v)
            ttnn.deallocate(w)
            o = ttnn.permute(o, (0, 2, 1))
            g = ttnn.linear(
                m,
                self.g_weight[:, i * self.head_dim : (i + 1) * self.head_dim],
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
            )
            o = ttnn.multiply(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            ttnn.deallocate(g)
            o = ttnn.linear(
                o,
                self.o_weight[i * self.head_dim : (i + 1) * self.head_dim, :],
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
            )
            o_out = o if o_out is None else ttnn.add(o_out, o)
        o_out = ttnn.reshape(o_out, (1, *o_out.shape))
        return o_out


class OuterProductMean(Module):
    def __init__(
        self,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.norm_weight = self.torch_to_tt("norm.weight")
        self.norm_bias = self.torch_to_tt("norm.bias")
        self.a_weight = self.torch_to_tt("proj_a.weight")
        self.b_weight = self.torch_to_tt("proj_b.weight")
        self.o_weight = self.torch_to_tt("proj_o.weight")
        self.o_bias = self.torch_to_tt("proj_o.bias")

    def __call__(self, x: ttnn.Tensor, msa_mask: ttnn.Tensor | None = None, n_msa: int | None = None) -> ttnn.Tensor:
        x = ttnn.reshape(x, tuple(x.shape)[1:])
        m = ttnn.layer_norm(
            x,
            weight=self.norm_weight,
            bias=self.norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        a = ttnn.linear(
            m,
            self.a_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        b = ttnn.linear(
            m,
            self.b_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        ttnn.deallocate(m)
        if msa_mask is not None:
            a = ttnn.multiply_(a, msa_mask)
        S, I, C = a.shape
        _, J, D = b.shape
        a = ttnn.permute(a, (1, 2, 0))  # (I, C, S)
        b = ttnn.permute(b, (2, 1, 0))
        b = ttnn.to_layout(b, ttnn.ROW_MAJOR_LAYOUT)
        b = ttnn.reshape(b, (-1, S))
        b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)
        if I > SEQ_LEN_MORE_CHUNKING:
            # Compact large tensors before OPM matmuls to reduce DRAM fragmentation.
            a = ttnn.reallocate(a)
            b = ttnn.reallocate(b)
        def outer_product_mean(a_in):
            rows = a_in.shape[0]
            a_flat = ttnn.reshape(a_in, (rows * C, S))
            z = ttnn.matmul(a_flat, b, transpose_b=True, compute_kernel_config=self.compute_kernel_config)
            ttnn.deallocate(a_flat)
            z = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
            z = ttnn.reshape(z, (rows, C * D, J))
            z = ttnn.to_layout(z, ttnn.TILE_LAYOUT)
            z = ttnn.permute(z, (0, 2, 1))
            z = ttnn.multiply_(z, 1 / (n_msa if n_msa is not None else S))
            out = ttnn.linear(
                z,
                self.o_weight,
                bias=self.o_bias,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
            )
            ttnn.deallocate(z)
            return out

        if I > SEQ_LEN_MORE_CHUNKING:
            z_acc = None
            for i in range(0, I, OPM_CHUNK_SIZE):
                part = outer_product_mean(a[i : min(i + OPM_CHUNK_SIZE, I), :, :])
                if z_acc is None:
                    z_acc = part
                else:
                    z_old = z_acc
                    z_acc = ttnn.concat([z_old, part], dim=0)
                    ttnn.deallocate(z_old)
                    ttnn.deallocate(part)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            z = z_acc
        else:
            z = outer_product_mean(a)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
        z = ttnn.reshape(z, (1, *z.shape))
        return z


class MSALayer(Module):
    def __init__(
        self,
        avg_head_dim: int,
        avg_n_heads: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.msa_transition = Transition(
            self.scope("msa_transition"), compute_kernel_config
        )
        self.pair_weighted_averaging = PairWeightedAveraging(
            head_dim=avg_head_dim,
            n_heads=avg_n_heads,
            state_dict=self.scope("pair_weighted_averaging"),
            compute_kernel_config=compute_kernel_config,
        )
        self.outer_product_mean = OuterProductMean(
            state_dict=self.scope("outer_product_mean"),
            compute_kernel_config=compute_kernel_config,
        )
        self.pairformer_layer = PairformerLayer(
            tri_att_head_dim,
            tri_att_n_heads,
            None,
            None,
            False,
            self.scope("pairformer_layer"),
            compute_kernel_config,
        )

    def __call__(
        self,
        z: ttnn.Tensor,
        m: ttnn.Tensor,
        mask: ttnn.Tensor | None,
        attn_mask: ttnn.Tensor | None,
        msa_mask: ttnn.Tensor | None,
        n_msa: int | None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        S = m.shape[2]
        if S > SEQ_LEN_MORE_CHUNKING:
            z = ttnn.reallocate(z)
            m_acc = None
            N = m.shape[1]
            for s in range(0, N, MSA_CHUNK_SIZE):
                mc = m[:, s:min(s + MSA_CHUNK_SIZE, N), :]
                mc = ttnn.add_(mc, self.pair_weighted_averaging(mc, z, attn_mask))
                mc = ttnn.add_(mc, self.msa_transition(mc))
                if m_acc is None:
                    m_acc = mc
                else:
                    m_old = m_acc
                    m_acc = ttnn.concat([m_old, mc], dim=1)
                    ttnn.deallocate(m_old)
                    ttnn.deallocate(mc)
            ttnn.deallocate(m)
            m = m_acc
            m = ttnn.reallocate(m)
            z = ttnn.add_(z, self.outer_product_mean(m, msa_mask, n_msa))
        else:
            m = ttnn.add_(m, self.pair_weighted_averaging(m, z, attn_mask))
            m = ttnn.add_(m, self.msa_transition(m))
            z = ttnn.add_(z, self.outer_product_mean(m, msa_mask, n_msa))

        z = self.pairformer_layer(
            None, z, mask=mask, attn_mask_start=attn_mask, attn_mask_end=attn_mask,
        )[1]

        return z, m


class MSA(Module):
    def __init__(
        self,
        n_blocks: int,
        avg_head_dim: int,
        avg_n_heads: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.s_weight = self.torch_to_tt("s_proj.weight")
        self.msa_weight = self.torch_to_tt("msa_proj.weight")
        self.blocks = [
            MSALayer(
                avg_head_dim,
                avg_n_heads,
                tri_att_head_dim,
                tri_att_n_heads,
                self.scope(f"layers.{i}"),
                compute_kernel_config,
            )
            for i in range(n_blocks)
        ]

    def __call__(
        self,
        z: ttnn.Tensor,
        m: ttnn.Tensor,
        emb: ttnn.Tensor,
        mask: ttnn.Tensor | None,
        attn_mask: ttnn.Tensor | None,
        msa_mask: ttnn.Tensor | None,
        n_msa: int | None,
    ) -> ttnn.Tensor:
        m = ttnn.linear(
            m,
            self.msa_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        m = ttnn.add_(
            m,
            ttnn.linear(
                emb,
                self.s_weight,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
            ),
        )
        for block in self.blocks:
            z, m = block(z, m, mask, attn_mask, msa_mask, n_msa)
        return z


class Diffusion(Module):
    def __init__(
        self,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self._s_conditioned = None
        self._c_reshaped = None
        self.conditioner_norm_weight = self.torch_to_tt(
            "single_conditioner.norm_single.weight"
        )
        self.conditioner_norm_bias = self.torch_to_tt(
            "single_conditioner.norm_single.bias"
        )
        self.conditioner_embed_weight = self.torch_to_tt(
            "single_conditioner.single_embed.weight"
        )
        self.conditioner_embed_bias = self.torch_to_tt(
            "single_conditioner.single_embed.bias"
        )
        self.conditioner_fourier_embed_weight = self.torch_to_tt(
            "single_conditioner.fourier_embed.proj.weight"
        )
        self.conditioner_fourier_embed_bias = self.torch_to_tt(
            "single_conditioner.fourier_embed.proj.bias"
        )
        self.conditioner_norm_fourier_weight = self.torch_to_tt(
            "single_conditioner.norm_fourier.weight"
        )
        self.conditioner_norm_fourier_bias = self.torch_to_tt(
            "single_conditioner.norm_fourier.bias"
        )
        self.conditioner_fourier_single_weight = self.torch_to_tt(
            "single_conditioner.fourier_to_single.weight"
        )
        self.conditioner_transition_0 = Transition(
            self.scope("single_conditioner.transitions.0"),
            compute_kernel_config,
        )
        self.conditioner_transition_1 = Transition(
            self.scope("single_conditioner.transitions.1"),
            compute_kernel_config,
        )
        self.r_to_q_weight = self.torch_to_tt(
            "atom_attention_encoder.r_to_q_trans.weight"
        )
        self.encoder = DiffusionTransformer(
            n_layers=ATOM_N_LAYERS,
            dim=ATOM_DIM,
            n_heads=ATOM_N_HEADS,
            atom_level=True,
            state_dict=self.scope("atom_attention_encoder.atom_encoder.diffusion_transformer"),
            compute_kernel_config=compute_kernel_config,
        )
        self.atom_to_token_weight = self.torch_to_tt(
            "atom_attention_encoder.atom_to_token_trans.0.weight"
        )
        self.s_to_a_norm_weight = self.torch_to_tt("s_to_a_linear.0.weight")
        self.s_to_a_norm_bias = self.torch_to_tt("s_to_a_linear.0.bias")
        self.s_to_a_linear_weight = self.torch_to_tt("s_to_a_linear.1.weight")
        self.token_transformer = DiffusionTransformer(
            n_layers=TOKEN_N_LAYERS,
            dim=TOKEN_DIM,
            n_heads=TOKEN_N_HEADS,
            atom_level=False,
            state_dict=self.scope("token_transformer"),
            compute_kernel_config=compute_kernel_config,
        )
        self.a_norm_weight = self.torch_to_tt("a_norm.weight")
        self.a_norm_bias = self.torch_to_tt("a_norm.bias")
        self.a_to_q_weight = self.torch_to_tt(
            "atom_attention_decoder.a_to_q_trans.weight"
        )
        self.decoder = DiffusionTransformer(
            n_layers=ATOM_N_LAYERS,
            dim=ATOM_DIM,
            n_heads=ATOM_N_HEADS,
            atom_level=True,
            state_dict=self.scope("atom_attention_decoder.atom_decoder.diffusion_transformer"),
            compute_kernel_config=compute_kernel_config,
        )
        self.feat_to_pos_norm_weight = self.torch_to_tt(
            "atom_attention_decoder.atom_feat_to_atom_pos_update.0.weight"
        )
        self.feat_to_pos_norm_bias = self.torch_to_tt(
            "atom_attention_decoder.atom_feat_to_atom_pos_update.0.bias"
        )
        self.feat_to_pos_linear_weight = self.torch_to_tt(
            "atom_attention_decoder.atom_feat_to_atom_pos_update.1.weight"
        )

    def __call__(
        self,
        r: ttnn.Tensor,
        times: ttnn.Tensor,
        s_inputs: ttnn.Tensor,
        s_trunk: ttnn.Tensor,
        q: ttnn.Tensor,
        c: ttnn.Tensor,
        bias_encoder: ttnn.Tensor,
        bias_token: ttnn.Tensor,
        bias_decoder: ttnn.Tensor,
        keys_indexing: ttnn.Tensor,
        atom_to_token: ttnn.Tensor,
        atom_to_token_normed: ttnn.Tensor,
        large_seq_len: bool = False,
    ) -> ttnn.Tensor:
        B, N, D = q.shape
        NW = N // ATOM_WINDOW
        if self._s_conditioned is None:
            s = ttnn.concat([s_trunk, s_inputs], dim=-1)
            s = ttnn.layer_norm(
                s,
                weight=self.conditioner_norm_weight,
                bias=self.conditioner_norm_bias,
                epsilon=1e-5,
                compute_kernel_config=self.compute_kernel_config,
            )
            self._s_conditioned = ttnn.linear(
                s,
                self.conditioner_embed_weight,
                bias=self.conditioner_embed_bias,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
            )
            ttnn.deallocate(s)
            self._c_reshaped = ttnn.reshape(c, (B, NW, ATOM_WINDOW, -1))
        r_to_q = ttnn.linear(
            r,
            self.r_to_q_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        q = ttnn.add(q, r_to_q)
        ttnn.deallocate(r_to_q)
        q = ttnn.reshape(q, (B, NW, ATOM_WINDOW, -1))
        q = self.encoder(
            q,
            self._c_reshaped,
            bias_encoder,
            keys_indexing,
            large_seq_len=large_seq_len,
        )
        q = ttnn.reshape(q, (B, NW * ATOM_WINDOW, D))
        a = ttnn.linear(
            q,
            self.atom_to_token_weight,
            compute_kernel_config=self.compute_kernel_config,
            activation="relu",
            core_grid=CORE_GRID_MAIN,
        )
        a = ttnn.matmul(
            a,
            atom_to_token_normed,
            transpose_a=True,
            compute_kernel_config=self.compute_kernel_config,
        )
        a = ttnn.permute(a, (0, 2, 1))
        times = ttnn.unsqueeze(times, 1)
        fourier = ttnn.linear(
            times,
            self.conditioner_fourier_embed_weight,
            bias=self.conditioner_fourier_embed_bias,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        fourier = ttnn.multiply(fourier, 2 * pi)
        fourier = ttnn.cos(fourier)
        fourier = ttnn.layer_norm(
            fourier,
            weight=self.conditioner_norm_fourier_weight,
            bias=self.conditioner_norm_fourier_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        fourier = ttnn.linear(
            fourier,
            self.conditioner_fourier_single_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        fourier = ttnn.unsqueeze(fourier, 1)
        s = ttnn.add(self._s_conditioned, fourier)
        ttnn.deallocate(fourier)
        s_update = self.conditioner_transition_0(s)
        s = ttnn.add(s, s_update)
        ttnn.deallocate(s_update)
        s_update = self.conditioner_transition_1(s)
        s = ttnn.add(s, s_update)
        ttnn.deallocate(s_update)
        s_to_a = ttnn.layer_norm(
            s,
            weight=self.s_to_a_norm_weight,
            bias=self.s_to_a_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        s_to_a = ttnn.linear(
            s_to_a,
            self.s_to_a_linear_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        a = ttnn.add(a, s_to_a)
        ttnn.deallocate(s_to_a)
        a = self.token_transformer(a, s, bias_token)
        ttnn.deallocate(s)
        a = ttnn.layer_norm(
            a,
            weight=self.a_norm_weight,
            bias=self.a_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        a_to_q = ttnn.linear(
            a,
            self.a_to_q_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        # TODO: Inline with transpose_a=True after newest tt-metal release.
        a_to_q = ttnn.permute(a_to_q, (0, 2, 1))
        a_to_q = ttnn.matmul(
            a_to_q,
            atom_to_token,
            transpose_b=True,
            compute_kernel_config=self.compute_kernel_config,
        )
        a_to_q = ttnn.permute(a_to_q, (0, 2, 1))
        q = ttnn.add(q, a_to_q)
        ttnn.deallocate(a_to_q)
        q = ttnn.reshape(q, (B, NW, ATOM_WINDOW, -1))
        q = self.decoder(
            q,
            self._c_reshaped,
            bias_decoder,
            keys_indexing,
            large_seq_len=large_seq_len,
        )
        q = ttnn.reshape(q, (B, NW * ATOM_WINDOW, D))
        r_update = ttnn.layer_norm(
            q,
            weight=self.feat_to_pos_norm_weight,
            bias=self.feat_to_pos_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        r_update = ttnn.linear(
            r_update,
            self.feat_to_pos_linear_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        ttnn.deallocate(q)
        return r_update


class TorchWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = None
        self.tt_device = get_device()
        self._runtime_cache = {}
        self._first_forward_pass = True
        self.compute_kernel_config = ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _from_torch(self, x: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
        return ttnn.from_torch(
            x,
            device=self.tt_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
        )

    def _to_torch(self, x: ttnn.Tensor) -> torch.Tensor:
        return torch.Tensor(ttnn.to_torch(x)).to(torch.float32)

    def _cache_set(self, key: str, value):
        self._runtime_cache[key] = value
        return value

    def _cache_get(self, key: str, default=None):
        return self._runtime_cache.get(key, default)

    def _cache_has_all(self, keys: tuple[str, ...]) -> bool:
        return all(key in self._runtime_cache for key in keys)

    def _deallocate_tensor_like(self, value):
        if value is None:
            return
        # Runtime caches may be a single TT tensor or small containers of TT tensors.
        if isinstance(value, (list, tuple)):
            for item in value:
                self._deallocate_tensor_like(item)
            return
        try:
            if isinstance(value, ttnn.Tensor):
                ttnn.deallocate(value)
        except Exception:
            # Best effort cleanup: stale/already-freed buffers should not break reset.
            pass

    def _clear_cached_attrs(self, obj, attr_names):
        for attr in attr_names:
            value = getattr(obj, attr, None)
            self._deallocate_tensor_like(value)
            setattr(obj, attr, None)

    def _clear_runtime_cache(self):
        for value in self._runtime_cache.values():
            self._deallocate_tensor_like(value)
        self._runtime_cache.clear()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.module = self._create_module(WeightScope.wrap(state_dict).child(prefix[:-1]))

    def _create_module(self, weights: WeightScope):
        raise NotImplementedError

    def reset_static_cache(self):
        """Reset cached static data so it is recomputed on the next forward pass.

        Call between proteins when input dimensions change.
        """
        self._clear_runtime_cache()
        self._first_forward_pass = True


class PairformerModule(TorchWrapper):
    def __init__(
        self,
        n_blocks: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        att_head_dim: int,
        att_n_heads: int,
        transform_s: bool,
        affinity: bool = False,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.tri_att_head_dim = tri_att_head_dim
        self.tri_att_n_heads = tri_att_n_heads
        self.att_head_dim = att_head_dim
        self.att_n_heads = att_n_heads
        self.transform_s = transform_s
        self.affinity = affinity

    def _create_module(self, weights: WeightScope):
        return Pairformer(
            self.n_blocks,
            self.tri_att_head_dim,
            self.tri_att_n_heads,
            self.att_head_dim,
            self.att_n_heads,
            self.transform_s,
            weights,
            self.compute_kernel_config,
            affinity=self.affinity,
        )

    def forward(
        self,
        s: torch.Tensor | None,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
        use_kernels: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        seq_len = z.shape[1]
        pad = (-seq_len) % PAIRFORMER_PAD_MULTIPLE

        required_cache_keys = ("mask_tt", "attn_mask_start_tt", "attn_mask_end_tt")
        if (not self._first_forward_pass) and (not self._cache_has_all(required_cache_keys)):
            self._clear_runtime_cache()
            self._first_forward_pass = True

        if pad:
            z = torch.nn.functional.pad(z, (0, 0, 0, pad, 0, pad))
            if s is not None:
                s = torch.nn.functional.pad(s, (0, 0, 0, pad))

        # Compute masks (once, reused across forward calls)
        if self._first_forward_pass:
            if self.affinity:
                # Affinity: cross-chain pair_mask, separate start/end additive masks
                if pad:
                    pair_mask = torch.nn.functional.pad(pair_mask, (0, pad, 0, pad))
                self._cache_set("mask_tt", self._from_torch(pair_mask))
                self._cache_set("attn_mask_start_tt", self._from_torch(pair_mask.permute(1, 0, 2).unsqueeze(2) * 1e9 - 1e9))
                self._cache_set("attn_mask_end_tt", self._from_torch(pair_mask.permute(2, 0, 1).unsqueeze(2) * 1e9 - 1e9))
            elif mask is not None or pad:
                # Non-affinity: 1D mask → additive [1,1,1,S], pair_mask [1,S,S] for TriangleMul
                mask_1d = mask if mask is not None else z.new_ones(1, seq_len)
                if pad:
                    mask_1d = torch.nn.functional.pad(mask_1d, (0, pad))
                    if pair_mask is not None:
                        pair_mask = torch.nn.functional.pad(pair_mask, (0, pad, 0, pad))
                self._cache_set("mask_tt", self._from_torch(pair_mask if pair_mask is not None else mask_1d))
                attn_mask = self._from_torch((1 - mask_1d).unsqueeze(1).unsqueeze(1) * -1e9)
                self._cache_set("attn_mask_start_tt", attn_mask)
                self._cache_set("attn_mask_end_tt", attn_mask)
            else:
                self._cache_set("mask_tt", None)
                self._cache_set("attn_mask_start_tt", None)
                self._cache_set("attn_mask_end_tt", None)
            self._first_forward_pass = False

        s_out, z_out = self.module(
            self._from_torch(s) if s is not None else None,
            self._from_torch(z),
            self._cache_get("mask_tt"),
            self._cache_get("attn_mask_start_tt"),
            self._cache_get("attn_mask_end_tt"),
        )

        s_result = self._to_torch(s_out)[:, :seq_len, :] if s_out is not None else None
        z_result = self._to_torch(z_out)[:, :seq_len, :seq_len, :]
        return s_result, z_result


class DiffusionModule(TorchWrapper):
    def __init__(self):
        super().__init__()

    def _create_module(self, weights: WeightScope):
        return Diffusion(weights, self.compute_kernel_config)

    def forward(
        self,
        r: torch.Tensor,
        times: torch.Tensor,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        q: torch.Tensor,
        c: torch.Tensor,
        bias_encoder: torch.Tensor,
        bias_token: torch.Tensor,
        bias_decoder: torch.Tensor,
        keys_indexing: torch.Tensor,
        mask: torch.Tensor,
        atom_to_token: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = q.shape
        NW = N // ATOM_WINDOW

        seq_len = s_inputs.shape[1]
        token_pad = (-seq_len) % PAIRFORMER_PAD_MULTIPLE
        padded_seq = seq_len + token_pad
        N_padded = padded_seq * MAX_ATOMS_PER_TOKEN
        assert N <= N_padded, f"N={N} exceeds max {N_padded} for padded_seq={padded_seq}. Increase MAX_ATOMS_PER_TOKEN."
        atom_pad = N_padded - N
        NW_padded = N_padded // ATOM_WINDOW
        K_padded = B * NW_padded

        required_cache_keys = (
            "s_inputs",
            "s_trunk",
            "q",
            "c",
            "keys_indexing",
            "bias_encoder",
            "bias_token",
            "bias_decoder",
            "atom_to_token",
            "atom_to_token_normed",
            "atom_pad",
        )
        if (not self._first_forward_pass) and (not self._cache_has_all(required_cache_keys)):
            self._clear_runtime_cache()
            self._first_forward_pass = True

        # Compute all static data once (everything except r and times is constant across diffusion steps)
        if self._first_forward_pass:
            if token_pad:
                s_inputs = torch.nn.functional.pad(s_inputs, (0, 0, 0, token_pad))
                s_trunk = torch.nn.functional.pad(s_trunk, (0, 0, 0, token_pad))
            self._cache_set("s_inputs", self._from_torch(s_inputs))
            self._cache_set("s_trunk", self._from_torch(s_trunk))

            q_pt = q if r.shape[0] == q.shape[0] else torch.repeat_interleave(q, r.shape[0], dim=0)
            c_pt = c if r.shape[0] == c.shape[0] else torch.repeat_interleave(c, r.shape[0], dim=0)
            if atom_pad:
                q_pt = torch.nn.functional.pad(q_pt, (0, 0, 0, atom_pad))
                c_pt = torch.nn.functional.pad(c_pt, (0, 0, 0, atom_pad))
            self._cache_set("q", self._from_torch(q_pt))
            self._cache_set("c", self._from_torch(c_pt))

            if atom_pad:
                ki_pad_rows = 2 * NW_padded - keys_indexing.shape[0]
                ki_pad_cols = 8 * NW_padded - keys_indexing.shape[1]
                keys_indexing = torch.nn.functional.pad(keys_indexing, (0, ki_pad_cols, 0, ki_pad_rows))
            keys_indexing_tt = self._from_torch(keys_indexing, dtype=ttnn.bfloat4_b)
            self._cache_set("keys_indexing", keys_indexing_tt)

            if atom_pad:
                mask = torch.nn.functional.pad(mask, (0, atom_pad))
            mask = self._from_torch(mask)
            mask = ttnn.reshape(mask, (2 * K_padded, ATOM_WINDOW // 2, -1))
            # TODO: Inline with transpose_a=True after newest tt-metal release.
            mask = ttnn.permute(mask, (1, 2, 0))
            mask = ttnn.matmul(
                mask,
                keys_indexing_tt,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=CORE_GRID_MAIN,
            )
            mask = ttnn.permute(mask, (2, 0, 1))
            mask = ttnn.reshape(mask, (K_padded, 1, 1, -1))
            # Additive mask: 0 → valid, -1e9 → padded (bfloat16 for -1e9 precision)
            mask = (-1 * mask + 1) * -1e9

            def prepare_atom_bias(bias_pt):
                if atom_pad:
                    bias_pt = torch.nn.functional.pad(bias_pt, (0, 0, 0, 0, 0, 0, 0, NW_padded - NW))
                bias = self._from_torch(bias_pt)
                bias = ttnn.reshape(bias, (B * NW_padded, ATOM_WINDOW, ATOM_DIM, -1))
                bias = ttnn.permute(bias, (0, 3, 1, 2))
                bias = ttnn.add_(bias, mask)
                return ttnn.multiply_(bias, ATOM_WINDOW ** 0.5)

            self._cache_set("bias_encoder", prepare_atom_bias(bias_encoder))
            self._cache_set("bias_decoder", prepare_atom_bias(bias_decoder))

            if token_pad:
                bias_token = torch.nn.functional.pad(bias_token, (0, 0, 0, token_pad, 0, token_pad))
            bias = self._from_torch(bias_token)
            bias = ttnn.multiply_(
                bias, (TOKEN_DIM / TOKEN_N_HEADS) ** 0.5
            )
            bias_token_tt = ttnn.permute(bias, (0, 3, 1, 2))
            if token_pad:
                # Fuse additive padding mask into token bias (bfloat16 for -1e9)
                seq_mask = torch.zeros(1, 1, 1, padded_seq)
                seq_mask[..., seq_len:] = -1e9
                bias_token_tt = ttnn.add_(bias_token_tt, self._from_torch(seq_mask))
            self._cache_set("bias_token", bias_token_tt)

            if atom_pad or token_pad:
                atom_to_token = torch.nn.functional.pad(atom_to_token, (0, token_pad, 0, atom_pad))
            atom_to_token_tt = self._from_torch(atom_to_token)
            self._cache_set("atom_to_token", atom_to_token_tt)
            atom_to_token_normed_tt = ttnn.multiply(
                atom_to_token_tt,
                ttnn.reciprocal(
                    ttnn.sum(atom_to_token_tt, dim=1, keepdim=True) + 1e-6
                ),
            )
            self._cache_set("atom_to_token_normed", atom_to_token_normed_tt)

            self._cache_set("atom_pad", atom_pad)
            self._first_forward_pass = False

        atom_pad_cached = self._cache_get("atom_pad", 0)
        if atom_pad_cached:
            r = torch.nn.functional.pad(r, (0, 0, 0, atom_pad_cached))

        result = self._to_torch(
            self.module(
                self._from_torch(r),
                self._from_torch(times),
                self._cache_get("s_inputs"),
                self._cache_get("s_trunk"),
                self._cache_get("q"),
                self._cache_get("c"),
                self._cache_get("bias_encoder"),
                self._cache_get("bias_token"),
                self._cache_get("bias_decoder"),
                self._cache_get("keys_indexing"),
                self._cache_get("atom_to_token"),
                self._cache_get("atom_to_token_normed"),
                large_seq_len=seq_len > SEQ_LEN_MORE_CHUNKING,
            )
        )
        result = result[:, :N, :]
        return result

    def reset_static_cache(self):
        super().reset_static_cache()
        if self.module is not None:
            self._clear_cached_attrs(self.module, ("_s_conditioned", "_c_reshaped"))
            for layer in self.module.encoder.layers + self.module.decoder.layers:
                self._clear_cached_attrs(layer, ("s_o",))


class MSAModule(TorchWrapper):
    def __init__(
        self,
        n_blocks: int,
        avg_head_dim: int,
        avg_n_heads: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.avg_head_dim = avg_head_dim
        self.avg_n_heads = avg_n_heads
        self.tri_att_head_dim = tri_att_head_dim
        self.tri_att_n_heads = tri_att_n_heads

    def _create_module(self, weights: WeightScope):
        return MSA(
            self.n_blocks,
            self.avg_head_dim,
            self.avg_n_heads,
            self.tri_att_head_dim,
            self.tri_att_n_heads,
            weights,
            self.compute_kernel_config,
        )

    def forward(
        self,
        z: torch.Tensor,
        emb: torch.Tensor,
        feats: dict[str, torch.Tensor],
        use_kernels: bool = False,
    ) -> torch.Tensor:
        m = torch.cat(
            [
                torch.nn.functional.one_hot(feats["msa"], num_classes=33),
                feats["has_deletion"].unsqueeze(-1),
                feats["deletion_value"].unsqueeze(-1),
                feats["msa_paired"].unsqueeze(-1),
            ],
            dim=-1,
        )

        seq_len = z.shape[1]
        n_msa = m.shape[1]
        seq_pad = (-seq_len) % PAIRFORMER_PAD_MULTIPLE
        msa_pad = (-n_msa) % MSA_PAD_MULTIPLE

        required_cache_keys = ("mask_tt", "attn_mask_tt", "msa_mask_tt", "n_msa")
        if (not self._first_forward_pass) and (not self._cache_has_all(required_cache_keys)):
            self._clear_runtime_cache()
            self._first_forward_pass = True

        if seq_pad:
            z = torch.nn.functional.pad(z, (0, 0, 0, seq_pad, 0, seq_pad))
            emb = torch.nn.functional.pad(emb, (0, 0, 0, seq_pad))
        if seq_pad or msa_pad:
            m = torch.nn.functional.pad(m, (0, 0, 0, seq_pad, 0, msa_pad))

        # Compute masks (once, reused across forward calls)
        if self._first_forward_pass:
            if seq_pad:
                padded_seq = seq_len + seq_pad
                mask_1d = z.new_ones(1, padded_seq)
                mask_1d[:, seq_len:] = 0.0
                # 2D mask for TriangleMultiplication (row + column masking)
                self._cache_set("mask_tt", self._from_torch(mask_1d.unsqueeze(-1) * mask_1d.unsqueeze(1)))
                # 4D additive mask for TriangleAttention (bfloat16 for -1e9)
                self._cache_set("attn_mask_tt", self._from_torch((1 - mask_1d).unsqueeze(1).unsqueeze(1) * -1e9))
            else:
                self._cache_set("mask_tt", None)
                self._cache_set("attn_mask_tt", None)
            if msa_pad:
                padded_msa = n_msa + msa_pad
                msa_mask = z.new_zeros(padded_msa, 1, 1)
                msa_mask[:n_msa] = 1.0
                self._cache_set("msa_mask_tt", self._from_torch(msa_mask))
                self._cache_set("n_msa", n_msa)
            else:
                self._cache_set("msa_mask_tt", None)
                self._cache_set("n_msa", None)
            self._first_forward_pass = False

        z_out = self._to_torch(
            self.module(
                self._from_torch(z),
                self._from_torch(m),
                self._from_torch(emb),
                self._cache_get("mask_tt"),
                self._cache_get("attn_mask_tt"),
                self._cache_get("msa_mask_tt"),
                self._cache_get("n_msa"),
            )
        )

        z_out = z_out[:, :seq_len, :seq_len, :]
        return z_out
