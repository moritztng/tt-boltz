import os
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
TRANSITION_BATCH_CHUNKING_THRESHOLD = 1024
TRANSITION_W_CHUNKING_THRESHOLD = 1024
TRANSITION_H_CHUNK_SIZE_FAST = 32
TRANSITION_H_CHUNK_SIZE = 16
_FAST_MODE = False
TRIANGLE_MULT_L1_MAX_SEQ_FAST = 640
TRIANGLE_MULT_L1_MAX_SEQ_FAST_13X10 = 704
TRIANGLE_MULT_L1_MAX_SEQ = 352
# Set by _apply_grid_thresholds: True on grids smaller than 11x10 (e.g. Wormhole).
# Tightens the L1-edge chunking thresholds and chunk sizes above this comment block.
_IS_SMALL_GRID = False
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

COMPUTE_GRID_X_11 = 11
COMPUTE_GRID_X_13 = 13
COMPUTE_GRID_Y = 10

CORE_GRID_MAIN = ttnn.CoreGrid(y=COMPUTE_GRID_Y, x=COMPUTE_GRID_X_11)
COMPUTE_GRID_MAIN = (CORE_GRID_MAIN.x, CORE_GRID_MAIN.y)

def _dtype():
    return ttnn.bfloat8_b if _FAST_MODE else ttnn.bfloat16


def _adaln_memory_config(atom_level: bool, large_seq_len: bool) -> ttnn.MemoryConfig | None:
    if not atom_level:
        return None
    return ttnn.DRAM_MEMORY_CONFIG if large_seq_len else ttnn.L1_MEMORY_CONFIG


def _triangle_mul_memory_config(seq_len: int) -> ttnn.MemoryConfig:
    if _FAST_MODE:
        l1_max_seq = (
            TRIANGLE_MULT_L1_MAX_SEQ_FAST_13X10
            if COMPUTE_GRID_MAIN[0] == COMPUTE_GRID_X_13
            else TRIANGLE_MULT_L1_MAX_SEQ_FAST
        )
    else:
        l1_max_seq = TRIANGLE_MULT_L1_MAX_SEQ
    return ttnn.L1_MEMORY_CONFIG if seq_len <= l1_max_seq else ttnn.DRAM_MEMORY_CONFIG


@lru_cache(maxsize=None)
def _sdpa_program_config(q_chunk_size: int, k_chunk_size: int) -> ttnn.SDPAProgramConfig:
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=COMPUTE_GRID_MAIN,
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
    gx, gy = COMPUTE_GRID_MAIN
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


def _apply_grid_thresholds(grid: tuple[int, int]) -> None:
    """Retune L1-edge thresholds and chunk sizes for grids smaller than the
    11x10 Blackhole baseline (e.g. Wormhole 8x8 has ~55% of its aggregate L1),
    so chunking kicks in early enough to avoid L1/CB clashes."""
    global _IS_SMALL_GRID, SEQ_LEN_MORE_CHUNKING, TRANSITION_BATCH_CHUNKING_THRESHOLD
    global TRANSITION_W_CHUNKING_THRESHOLD, TRIANGLE_ATT_CHUNK_SIZE_FAST
    global TRANSITION_W_CHUNK_SIZE, TRIANGLE_MULT_L1_MAX_SEQ_FAST
    _IS_SMALL_GRID = grid[0] * grid[1] < COMPUTE_GRID_X_11 * COMPUTE_GRID_Y
    if not _IS_SMALL_GRID:
        return  # Keep Blackhole baseline values
    SEQ_LEN_MORE_CHUNKING = 640
    TRANSITION_BATCH_CHUNKING_THRESHOLD = 640
    TRANSITION_W_CHUNKING_THRESHOLD = 640
    TRIANGLE_ATT_CHUNK_SIZE_FAST = 512
    TRANSITION_W_CHUNK_SIZE = 512
    TRIANGLE_MULT_L1_MAX_SEQ_FAST = 320  # half of 640, snapped to TRIANGLE_MULT_CHUNK_SIZE


def _configure_active_compute_grid(device: ttnn.Device) -> None:
    """Snap to a tuned 13x10 or 11x10 Blackhole grid when available; on smaller
    archs (e.g. Wormhole B0 8x8 with ETH dispatch) adopt the device's grid."""
    global CORE_GRID_MAIN, COMPUTE_GRID_MAIN

    gx, gy = COMPUTE_GRID_X_11, COMPUTE_GRID_Y
    try:
        a = device.compute_with_storage_grid_size()
        ax, ay = int(a.x), int(a.y)
        if ax >= COMPUTE_GRID_X_13:
            gx = COMPUTE_GRID_X_13
        elif ax < COMPUTE_GRID_X_11 or ay < COMPUTE_GRID_Y:
            gx, gy = ax, ay
    except Exception:
        pass

    if (gx, gy) == COMPUTE_GRID_MAIN:
        return

    CORE_GRID_MAIN = ttnn.CoreGrid(y=gy, x=gx)
    COMPUTE_GRID_MAIN = (gx, gy)
    _apply_grid_thresholds((gx, gy))
    _sdpa_program_config.cache_clear()
    _sdpa_program_config_for_lengths.cache_clear()
    _triangle_mul_program_config.cache_clear()


def set_fast_mode(enabled: bool) -> None:
    """Set fast block-fp8 mode for the current worker process."""
    global _FAST_MODE
    _FAST_MODE = bool(enabled)


_device = None

def get_device():
    """Open (or return cached) TT device 0.

    Worker processes set TT_VISIBLE_DEVICES before importing ttnn, so the
    assigned physical chip appears as logical device 0.
    """
    global _device
    if _device is None:
        device_id = int(os.environ.get("TT_BOLTZ_LOGICAL_DEVICE_ID", "0"))
        # Wormhole: dispatch on Ethernet cores so the full 8x8 Tensix grid
        # (rather than 8x7 after worker-dispatch reservation) is available.
        kwargs = (
            {"dispatch_core_config": ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH)}
            if ttnn.get_arch_name() == "wormhole_b0" else {}
        )
        _device = ttnn.open_device(device_id=device_id, **kwargs)
        _configure_active_compute_grid(_device)
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

    def _lin(self, x, w, **kw):
        return ttnn.linear(x, w, compute_kernel_config=self.compute_kernel_config,
                           core_grid=CORE_GRID_MAIN, **kw)


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
            core_grid=CORE_GRID_MAIN,
        )
        ttnn.deallocate(x)
        g_out = ttnn.linear(
            x_norm_in,
            self.g_out_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=_dtype(),
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
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
            core_grid=CORE_GRID_MAIN,
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
                core_grid=CORE_GRID_MAIN,
            )
            ttnn.deallocate(o_in)
            return x_out

        S = x.shape[0]
        need_chunk = S > SEQ_LEN_MORE_CHUNKING and (self.affinity or not _FAST_MODE or _IS_SMALL_GRID)
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
                    core_grid=CORE_GRID_MAIN,
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
                core_grid=CORE_GRID_MAIN,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(x)
            return x_dram
        if len(x.shape) < 4:
            batch_chunking_threshold = (
                SEQ_LEN_MORE_CHUNKING
                if COMPUTE_GRID_MAIN[0] == COMPUTE_GRID_X_13
                else TRANSITION_BATCH_CHUNKING_THRESHOLD
            )
            if x.shape[1] > batch_chunking_threshold:
                return ttnn.concat([swiglu(x[b:b+1, :, :]) for b in range(x.shape[0])], dim=0)
            return swiglu(x)

        H, W = x.shape[1], x.shape[2]
        transition_h_chunk_size = TRANSITION_H_CHUNK_SIZE_FAST if _FAST_MODE else TRANSITION_H_CHUNK_SIZE
        transition_w_chunking_threshold = (
            SEQ_LEN_MORE_CHUNKING
            if COMPUTE_GRID_MAIN[0] == COMPUTE_GRID_X_13
            else TRANSITION_W_CHUNKING_THRESHOLD
        )
        chunks = ttnn.chunk(x, -(-H // transition_h_chunk_size), dim=1)
        if W <= transition_w_chunking_threshold:
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


class MiniTriangularUpdate(Module):
    """Bi-directional triangular multiplicative update (BoltzGen Miniformer).

    Equivalent to PyTorch reference (boltzgen/.../triangular.py:MiniTriangularUpdate):

        x = norm_in(x)
        x = p_in(x) * sigmoid(g_in(x))        # (B, N, N, D)
        x = x * mask.unsqueeze(-1)
        a1, b1, a2, b2 = chunk(x, 4, dim=-1)  # 4 x (B, N, N, D/4)
        x1 = einsum("bikd,bjkd->bijd", a1, b1)  # outgoing-style
        x2 = einsum("bkid,bkjd->bijd", a2, b2)  # incoming-style
        x = cat([x1, x2], -1)                 # (B, N, N, D/2)
        x = norm_out(x)
        return p_out(x) * sigmoid(g_out(x))   # (B, N, N, D)

    Each einsum decomposes to a permute-matmul-permute, reusing the same
    permutation pattern as TriangleMultiplication (outgoing=False / incoming=True).
    """

    def __init__(
        self,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.norm_in_weight = self.torch_to_tt("norm_in.weight")
        self.norm_in_bias = self.torch_to_tt("norm_in.bias")
        self.p_in_weight = self.torch_to_tt("p_in.weight")
        self.g_in_weight = self.torch_to_tt("g_in.weight")
        self.norm_out_weight = self.torch_to_tt("norm_out.weight")
        self.norm_out_bias = self.torch_to_tt("norm_out.bias")
        self.p_out_weight = self.torch_to_tt("p_out.weight")
        self.g_out_weight = self.torch_to_tt("g_out.weight")

    @staticmethod
    def _matmul_einsum(
        a: ttnn.Tensor,
        b: ttnn.Tensor,
        ending: bool,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
        memory_config: ttnn.MemoryConfig,
    ) -> ttnn.Tensor:
        """Compute the einsum bikd,bjkd->bijd (ending=False, outgoing) or
        bkid,bkjd->bijd (ending=True, incoming) via permute-matmul-permute."""
        a_perm = (0, 3) + ((2, 1) if ending else (1, 2))
        b_perm = (0, 3) + ((1, 2) if ending else (2, 1))
        ap = ttnn.permute(a, a_perm, memory_config=memory_config)
        bp = ttnn.permute(b, b_perm, memory_config=memory_config)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        out = ttnn.matmul(
            ap,
            bp,
            compute_kernel_config=compute_kernel_config,
            memory_config=memory_config,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(ap)
        ttnn.deallocate(bp)
        return ttnn.permute(out, (0, 2, 3, 1), memory_config=memory_config)

    def __call__(self, x: ttnn.Tensor, mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        seq_len = x.shape[1]
        memory_config = _triangle_mul_memory_config(seq_len)

        x_norm = ttnn.layer_norm(
            x,
            weight=self.norm_in_weight,
            bias=self.norm_in_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        p = ttnn.linear(
            x_norm,
            self.p_in_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            core_grid=CORE_GRID_MAIN,
        )
        g = ttnn.linear(
            x_norm,
            self.g_in_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            core_grid=CORE_GRID_MAIN,
        )
        ttnn.deallocate(x_norm)
        x_gated = ttnn.multiply_(
            p, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]
        )
        ttnn.deallocate(g)
        if mask is not None:
            x_gated = ttnn.multiply_(x_gated, ttnn.unsqueeze(mask, -1))

        a1, b1, a2, b2 = ttnn.chunk(x_gated, chunks=4, dim=-1)
        ttnn.deallocate(x_gated)

        x1 = self._matmul_einsum(
            a1, b1, ending=False,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
        )
        x2 = self._matmul_einsum(
            a2, b2, ending=True,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
        )
        x = ttnn.concat([x1, x2], dim=-1)
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)

        x = ttnn.layer_norm(
            x,
            weight=self.norm_out_weight,
            bias=self.norm_out_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        p_out = ttnn.linear(
            x,
            self.p_out_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            core_grid=CORE_GRID_MAIN,
        )
        g_out = ttnn.linear(
            x,
            self.g_out_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            core_grid=CORE_GRID_MAIN,
        )
        ttnn.deallocate(x)
        return ttnn.multiply_(
            p_out, g_out, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]
        )


class MiniformerLayer(Module):
    """One Miniformer block: triangular + attention on s + transitions."""

    def __init__(
        self,
        att_head_dim: int,
        att_n_heads: int,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.triangular = MiniTriangularUpdate(
            self.scope("triangular"), compute_kernel_config
        )
        self.transition_z = Transition(self.scope("transition_z"), compute_kernel_config)
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
        self.transition_s = Transition(self.scope("transition_s"), compute_kernel_config)

    def __call__(
        self,
        s: ttnn.Tensor,
        z: ttnn.Tensor,
        mask: ttnn.Tensor | None = None,
        seq_mask: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        z_update = self.triangular(z, mask)
        z = ttnn.add_(z, z_update)
        ttnn.deallocate(z_update)

        z_update = self.transition_z(z)
        z = ttnn.add_(z, z_update)
        ttnn.deallocate(z_update)

        s_norm = ttnn.layer_norm(
            s,
            weight=self.pre_norm_s_weight,
            bias=self.pre_norm_s_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        s_update = self.attention_pair_bias(s_norm, z, seq_mask=seq_mask)
        ttnn.deallocate(s_norm)
        s = ttnn.add_(s, s_update)
        ttnn.deallocate(s_update)

        s_update = self.transition_s(s)
        s = ttnn.add_(s, s_update)
        ttnn.deallocate(s_update)
        return s, z


class Miniformer(Module):
    def __init__(
        self,
        n_blocks: int,
        att_head_dim: int,
        att_n_heads: int,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.blocks = [
            MiniformerLayer(
                att_head_dim,
                att_n_heads,
                self.scope(f"layers.{i}"),
                compute_kernel_config,
            )
            for i in range(n_blocks)
        ]

    def __call__(
        self,
        s: ttnn.Tensor,
        z: ttnn.Tensor,
        mask: ttnn.Tensor | None = None,
        seq_mask: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        for block in self.blocks:
            s, z = block(s, z, mask, seq_mask)
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
        self._s_o_src = None
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
        # s_o = sigmoid(linear(s)) is constant while s is (the atom conditioning
        # is reused across the 200 diffusion steps), so cache it — but only while
        # the same s object is fed. A new protein passes a fresh s, which rebuilds
        # it; this keeps the cache correct across proteins of any size without an
        # external reset (s for the atom encoder/decoder is _c_reshaped; for the
        # input-embedder atom transformer it is a fresh per-protein tensor).
        if self.atom_level and self.s_o is not None and self._s_o_src is s:
            s_o = self.s_o
        else:
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
                self._s_o_src = s
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
        self._cond_src = None
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
        # The conditioning (s_conditioned/c_reshaped) is constant across the 200
        # diffusion steps but changes per protein. Rebuild it whenever a new c
        # arrives (the wrapper passes a fresh c object per protein); reuse it
        # while the same c is fed step after step.
        if self._s_conditioned is None or self._cond_src is not c:
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
            self._cond_src = c
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
        # Keep explicit 3D axis reorder for this batched path; operand swapping
        # changed semantics under stricter matmul batch validation.
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
        kernel_cls = (
            ttnn.types.WormholeComputeKernelConfig
            if self.tt_device.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig
        )
        self.compute_kernel_config = kernel_cls(
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

    def _clear_runtime_cache(self):
        for value in self._runtime_cache.values():
            self._deallocate_tensor_like(value)
        self._runtime_cache.clear()

    def _load_from_state_dict(self, state_dict, prefix, _local_metadata, _strict, _missing_keys, _unexpected_keys, _error_msgs):
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


class MiniformerModule(TorchWrapper):
    """Public wrapper for BoltzGen's Miniformer (design-stage pairformer).

    Same interface as PairformerModule.forward(s, z, mask, pair_mask, ...) but
    drives the lighter Miniformer stack: one MiniTriangularUpdate per layer
    instead of 4 triangular ops.
    """

    def __init__(self, n_blocks: int, att_head_dim: int, att_n_heads: int):
        super().__init__()
        self.n_blocks = n_blocks
        self.att_head_dim = att_head_dim
        self.att_n_heads = att_n_heads

    def _create_module(self, weights: WeightScope):
        return Miniformer(
            self.n_blocks,
            self.att_head_dim,
            self.att_n_heads,
            weights,
            self.compute_kernel_config,
        )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
        use_kernels: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = z.shape[1]
        pad = (-seq_len) % PAIRFORMER_PAD_MULTIPLE

        required_cache_keys = ("mask_tt", "seq_mask_tt")
        if (not self._first_forward_pass) and (not self._cache_has_all(required_cache_keys)):
            self._clear_runtime_cache()
            self._first_forward_pass = True

        if pad:
            z = torch.nn.functional.pad(z, (0, 0, 0, pad, 0, pad))
            s = torch.nn.functional.pad(s, (0, 0, 0, pad))

        if self._first_forward_pass:
            mask_1d = mask if mask is not None else z.new_ones(1, seq_len)
            if pad:
                mask_1d = torch.nn.functional.pad(mask_1d, (0, pad))
                if pair_mask is not None:
                    pair_mask = torch.nn.functional.pad(pair_mask, (0, pad, 0, pad))
            # 2D pair-mask if provided, otherwise the 1D token mask (Miniformer
            # masks the bi-directional update by token, not by pair).
            self._cache_set(
                "mask_tt",
                self._from_torch(pair_mask if pair_mask is not None else mask_1d),
            )
            self._cache_set(
                "seq_mask_tt",
                self._from_torch((1 - mask_1d).unsqueeze(1).unsqueeze(1) * -1e9),
            )
            self._first_forward_pass = False

        s_out, z_out = self.module(
            self._from_torch(s),
            self._from_torch(z),
            self._cache_get("mask_tt"),
            self._cache_get("seq_mask_tt"),
        )

        s_result = self._to_torch(s_out)[:, :seq_len, :]
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

        # Compute all static data once (everything except r and times is constant
        # across diffusion steps).
        if self._first_forward_pass:
            # Device-resident conditioning chaining: q/c/biases/s_inputs/s_trunk may
            # arrive as ttnn tensors (already on device, no host upload). In that
            # case pad on device (concat with zeros, matching pad_dev2) and skip
            # _from_torch; otherwise the original host path. _dt = "is ttnn".
            def _dt(x):
                return isinstance(x, ttnn.Tensor)

            def _padcat(x, n, dim):
                if n == 0:
                    return x
                shp = list(x.shape); shp[dim] = n
                return ttnn.concat([x, self._from_torch(torch.zeros(shp))], dim=dim)

            if _dt(s_inputs):
                self._cache_set("s_inputs", _padcat(s_inputs, token_pad, 1))
                self._cache_set("s_trunk", _padcat(s_trunk, token_pad, 1))
            else:
                if token_pad:
                    s_inputs = torch.nn.functional.pad(s_inputs, (0, 0, 0, token_pad))
                    s_trunk = torch.nn.functional.pad(s_trunk, (0, 0, 0, token_pad))
                self._cache_set("s_inputs", self._from_torch(s_inputs))
                self._cache_set("s_trunk", self._from_torch(s_trunk))

            if _dt(q):
                # multiplicity==1 assumed for device chaining (no repeat_interleave)
                self._cache_set("q", _padcat(q, atom_pad, 1))
                self._cache_set("c", _padcat(c, atom_pad, 1))
            else:
                q_pt = q if r.shape[0] == q.shape[0] else torch.repeat_interleave(q, r.shape[0], dim=0)
                c_pt = c if r.shape[0] == c.shape[0] else torch.repeat_interleave(c, r.shape[0], dim=0)
                if atom_pad:
                    q_pt = torch.nn.functional.pad(q_pt, (0, 0, 0, atom_pad))
                    c_pt = torch.nn.functional.pad(c_pt, (0, 0, 0, atom_pad))
                self._cache_set("q", self._from_torch(q_pt))
                self._cache_set("c", self._from_torch(c_pt))

            # keys_indexing is a pure function of the windowing shape (K, W, H), so
            # it is identical across all same-shape proteins — reuse the cached one
            # on refresh (also avoids an unsupported bfloat4_b in-place copy).
            keys_indexing_tt = self._cache_get("keys_indexing")
            if keys_indexing_tt is None:
                if atom_pad:
                    ki_pad_rows = 2 * NW_padded - keys_indexing.shape[0]
                    ki_pad_cols = 8 * NW_padded - keys_indexing.shape[1]
                    keys_indexing = torch.nn.functional.pad(keys_indexing, (0, ki_pad_cols, 0, ki_pad_rows))
                keys_indexing_tt = self._cache_set("keys_indexing", self._from_torch(keys_indexing, dtype=ttnn.bfloat4_b))

            if atom_pad:
                mask = torch.nn.functional.pad(mask, (0, atom_pad))
            mask = self._from_torch(mask)
            mask = ttnn.reshape(mask, (2 * K_padded, ATOM_WINDOW // 2, -1))
            # transpose_a swaps only the last two dims and cannot replace this 3D axis reorder.
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
                if _dt(bias_pt):  # device: pad NW dim by concat, skip from_torch
                    bias = _padcat(bias_pt, NW_padded - NW, 1)
                else:
                    if atom_pad:
                        bias_pt = torch.nn.functional.pad(bias_pt, (0, 0, 0, 0, 0, 0, 0, NW_padded - NW))
                    bias = self._from_torch(bias_pt)
                bias = ttnn.reshape(bias, (B * NW_padded, ATOM_WINDOW, ATOM_DIM, -1))
                bias = ttnn.permute(bias, (0, 3, 1, 2))
                bias = ttnn.add_(bias, mask)
                return ttnn.multiply_(bias, ATOM_WINDOW ** 0.5)

            self._cache_set("bias_encoder", prepare_atom_bias(bias_encoder))
            self._cache_set("bias_decoder", prepare_atom_bias(bias_decoder))

            if _dt(bias_token):
                bias = _padcat(_padcat(bias_token, token_pad, 1), token_pad, 2)
            else:
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
            atom_to_token_tt = self._cache_set("atom_to_token", self._from_torch(atom_to_token))
            atom_to_token_normed_tt = ttnn.multiply(
                atom_to_token_tt,
                ttnn.reciprocal(
                    ttnn.sum(atom_to_token_tt, dim=1, keepdim=True) + 1e-6
                ),
            )
            self._cache_set("atom_to_token_normed", atom_to_token_normed_tt)

            self._cache_set("atom_pad", atom_pad)
            self._cache_set("large_seq_len", seq_len > SEQ_LEN_MORE_CHUNKING)
            self._cache_set("N_padded", N_padded)
            self._cache_set("N_real", N)
            self._first_forward_pass = False

        atom_pad_cached = self._cache_get("atom_pad", 0)
        if atom_pad_cached:
            r = torch.nn.functional.pad(r, (0, 0, 0, atom_pad_cached))

        result_tt = self.forward_device(self._from_torch(r), self._from_torch(times))

        result = self._to_torch(result_tt)
        result = result[:, :N, :]
        return result

    def forward_device(self, r_tt: ttnn.Tensor, times_tt: ttnn.Tensor) -> ttnn.Tensor:
        """Device-level score-model forward on the cached static conditioning.

        Assumes the static cache has been built (via a prior forward()/prepare).
        ``r_tt`` is the padded atom coords [B, N_padded, 3]; returns the atom
        coords update [B, N_padded, 3]. No host transfers — usable inside a trace.
        """
        return self.module(
            r_tt,
            times_tt,
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
            large_seq_len=self._cache_get("large_seq_len"),
        )


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


class TrunkRecycle:
    """Device-resident trunk recycle glue.

    Computes, entirely on the TT device::

        s = s_init + s_recycle(s_norm(s))
        z = z_init + z_recycle(z_norm(z))

    mirroring the torch ops in ``Boltz2.forward`` (boltz2.py:5197-5198).
    ``s_norm``/``z_norm`` are ``nn.LayerNorm`` (weight + bias, eps 1e-5);
    ``s_recycle``/``z_recycle`` are ``nn.Linear(.., bias=False)``. The ttnn
    weights are built directly from the already-loaded torch modules so this
    needs no separate state-dict load.
    """

    def __init__(self, s_norm, z_norm, s_recycle, z_recycle, compute_kernel_config):
        self.compute_kernel_config = compute_kernel_config
        device = get_device()

        def w(tensor, transpose=False):
            t = tensor.detach()
            if transpose:
                t = t.t().contiguous()
            return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

        self.s_norm_weight = w(s_norm.weight)
        self.s_norm_bias = w(s_norm.bias)
        self.z_norm_weight = w(z_norm.weight)
        self.z_norm_bias = w(z_norm.bias)
        # nn.Linear stores weight as [out, in]; ttnn.linear wants [in, out].
        self.s_recycle_weight = w(s_recycle.weight, transpose=True)
        self.z_recycle_weight = w(z_recycle.weight, transpose=True)

    def _branch(self, x, norm_weight, norm_bias, recycle_weight, init):
        x_norm = ttnn.layer_norm(
            x,
            weight=norm_weight,
            bias=norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        x_rec = ttnn.linear(
            x_norm,
            recycle_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN,
        )
        ttnn.deallocate(x_norm)
        out = ttnn.add(init, x_rec)
        ttnn.deallocate(x_rec)
        return out

    def __call__(self, s, z, s_init, z_init):
        s_out = self._branch(s, self.s_norm_weight, self.s_norm_bias, self.s_recycle_weight, s_init)
        z_out = self._branch(z, self.z_norm_weight, self.z_norm_bias, self.z_recycle_weight, z_init)
        return s_out, z_out


class TrunkModule(TorchWrapper):
    """Device-resident Boltz2 trunk (recycling) loop.

    Replaces the host-side recycling loop in ``Boltz2.forward``
    (boltz2.py:5193-5238) for the simplest case (no templates). The whole
    loop runs on the TT device: ``s``/``z`` are uploaded once as zeros, all
    per-protein constants (``s_init``/``z_init``/``s_inputs``, the MSA feature
    tensor and every mask) are built on host once and uploaded once, and only
    the final ``s``/``z`` come back to torch. This removes the per-iteration
    host round-trips that previously defeated ttnn tracing.

    It reuses the *inner* (already device-resident) ``MSA`` and ``Pairformer``
    modules owned by the existing ``MSAModule`` / ``PairformerModule`` wrappers,
    plus a ``TrunkRecycle`` for the glue. The mask / MSA-feature construction
    below mirrors ``MSAModule.forward`` and ``PairformerModule.forward`` exactly
    (kept in sync by the equivalence test in tests/).
    """

    def __init__(self, recycle: TrunkRecycle, msa_inner: "MSA", pairformer_inner: "Pairformer"):
        super().__init__()
        self.recycle = recycle
        self.msa = msa_inner
        self.pairformer = pairformer_inner

    def _build_static(self, s_inputs, s_init, z_init, feats):
        """Build + upload (once per protein) all loop-invariant device tensors.

        Returns a dict cached in ``self._runtime_cache`` and reused across the
        recycling iterations (and, later, across traced replays).
        """
        seq_len = z_init.shape[1]
        seq_pad = (-seq_len) % PAIRFORMER_PAD_MULTIPLE
        padded_seq = seq_len + seq_pad

        # ---- MSA feature tensor (host), mirrors MSAModule.forward ----
        m = torch.cat(
            [
                torch.nn.functional.one_hot(feats["msa"], num_classes=33),
                feats["has_deletion"].unsqueeze(-1),
                feats["deletion_value"].unsqueeze(-1),
                feats["msa_paired"].unsqueeze(-1),
            ],
            dim=-1,
        )
        n_msa = m.shape[1]
        msa_pad = (-n_msa) % MSA_PAD_MULTIPLE

        # ---- pad the per-protein constants ----
        pad = torch.nn.functional.pad
        s_init_p = pad(s_init, (0, 0, 0, seq_pad)) if seq_pad else s_init
        z_init_p = pad(z_init, (0, 0, 0, seq_pad, 0, seq_pad)) if seq_pad else z_init
        s_inputs_p = pad(s_inputs, (0, 0, 0, seq_pad)) if seq_pad else s_inputs
        m_p = pad(m, (0, 0, 0, seq_pad, 0, msa_pad)) if (seq_pad or msa_pad) else m

        # ---- Pairformer masks (mirror PairformerModule.forward, non-affinity) ----
        token_mask = feats["token_pad_mask"].float()
        pair_mask = token_mask[:, :, None] * token_mask[:, None, :]
        mask_1d_pf = token_mask
        if seq_pad:
            mask_1d_pf = pad(mask_1d_pf, (0, seq_pad))
            pair_mask = pad(pair_mask, (0, seq_pad, 0, seq_pad))
        pf_mask_tt = self._from_torch(pair_mask)
        pf_attn_tt = self._from_torch((1 - mask_1d_pf).unsqueeze(1).unsqueeze(1) * -1e9)

        # ---- MSA masks (mirror MSAModule.forward: derived from padding only) ----
        if seq_pad:
            mask_1d_msa = z_init.new_ones(1, padded_seq)
            mask_1d_msa[:, seq_len:] = 0.0
            msa_mask_tt = self._from_torch(mask_1d_msa.unsqueeze(-1) * mask_1d_msa.unsqueeze(1))
            msa_attn_tt = self._from_torch((1 - mask_1d_msa).unsqueeze(1).unsqueeze(1) * -1e9)
        else:
            msa_mask_tt = None
            msa_attn_tt = None
        if msa_pad:
            padded_msa = n_msa + msa_pad
            msa_row = z_init.new_zeros(padded_msa, 1, 1)
            msa_row[:n_msa] = 1.0
            msa_rowmask_tt = self._from_torch(msa_row)
            n_msa_arg = n_msa
        else:
            msa_rowmask_tt = None
            n_msa_arg = None

        static = {
            "seq_len": seq_len,
            "s_init_tt": self._from_torch(s_init_p),
            "z_init_tt": self._from_torch(z_init_p),
            "emb_tt": self._from_torch(s_inputs_p),
            "m_tt": self._from_torch(m_p),
            "pf_mask_tt": pf_mask_tt,
            "pf_attn_tt": pf_attn_tt,
            "msa_mask_tt": msa_mask_tt,
            "msa_attn_tt": msa_attn_tt,
            "msa_rowmask_tt": msa_rowmask_tt,
            "n_msa_arg": n_msa_arg,
        }
        for k, v in static.items():
            self._cache_set(k, v)
        return static

    def _iteration(self, s, z, st):
        """Run one recycling iteration fully on device; returns (s, z)."""
        # s = s_init + s_recycle(s_norm(s)); z = z_init + z_recycle(z_norm(z))
        s_rec, z_rec = self.recycle(s, z, st["s_init_tt"], st["z_init_tt"])
        ttnn.deallocate(s)
        ttnn.deallocate(z)

        # z = z + msa(z). The inner MSA mutates its z argument in place, so clone
        # z_rec first to preserve it for the residual add (matches the wrapper,
        # which passes a fresh upload each call).
        z_for_msa = ttnn.clone(z_rec)
        z_msa = self.msa(
            z_for_msa,
            st["m_tt"],
            st["emb_tt"],
            st["msa_mask_tt"],
            st["msa_attn_tt"],
            st["msa_rowmask_tt"],
            st["n_msa_arg"],
        )
        z = ttnn.add(z_rec, z_msa)
        ttnn.deallocate(z_rec)
        ttnn.deallocate(z_msa)

        # s, z = pairformer(s, z) -- inner mutates s_rec / z in place and returns them.
        s, z = self.pairformer(s_rec, z, st["pf_mask_tt"], st["pf_attn_tt"], st["pf_attn_tt"])
        return s, z

    def forward(self, s_inputs, s_init, z_init, feats, recycling_steps):
        st = self._build_static(s_inputs, s_init, z_init, feats)
        seq_len = st["seq_len"]

        s, z = self._run_eager(st, s_init.dtype, z_init.dtype, recycling_steps)

        # Device-resident chaining: stash an unpadded device clone of z (the 64MB
        # [1,N,N,128] pair tensor) so distogram/diffusion_conditioning reuse it
        # without a host re-upload. The caller frees it before the diffusion.
        self._dev_z = None
        if getattr(self, "_keep_device_z", False):
            # clone() -> a stable standalone buffer; slice() alone can alias z's
            # storage, which the ttnn.deallocate(z) below would then free.
            self._dev_z = ttnn.clone(
                ttnn.slice(z, [0, 0, 0, 0], [1, seq_len, seq_len, z.shape[-1]]))

        s_out = self._to_torch(s)[:, :seq_len, :]
        z_out = self._to_torch(z)[:, :seq_len, :seq_len, :]
        ttnn.deallocate(s)
        ttnn.deallocate(z)
        return s_out, z_out

    def _run_eager(self, st, s_dtype, z_dtype, recycling_steps):
        s = self._from_torch(torch.zeros(list(st["s_init_tt"].shape), dtype=s_dtype))
        z = self._from_torch(torch.zeros(list(st["z_init_tt"].shape), dtype=z_dtype))
        for _ in range(recycling_steps + 1):
            s, z = self._iteration(s, z, st)
        return s, z


# ---------------------------------------------------------------------------
# On-device weighted rigid (Kabsch) alignment.
#
# Replaces boltz2.weighted_rigid_align's torch SVD with a matmul-only Newton-
# Schulz polar decomposition so the diffusion reverse-alignment can run on the
# TT device (ttnn has no svd/eig/qr). For coords of the same molecule (noisy =
# denoised + noise, identical chirality) the optimal alignment is a proper
# rotation (det +1), so the orthogonal polar factor equals the Kabsch rotation
# and no reflection correction is needed. Validated against the torch SVD
# reference to ~1e-5 (see scripts/verify_kabsch_tt.py).
# ---------------------------------------------------------------------------

NEWTON_SCHULZ_ITERS = 8


def weighted_rigid_align_tt(
    true_coords: ttnn.Tensor,      # [B, N, 3]
    pred_coords: ttnn.Tensor,      # [B, N, 3]
    weights: ttnn.Tensor,          # [B, N, 1]
    compute_kernel_config,
    iters: int = NEWTON_SCHULZ_ITERS,
) -> ttnn.Tensor:
    """Align true_coords onto pred_coords (weighted Kabsch), entirely on device.

    Mirrors boltz2.weighted_rigid_align(true, pred, weights, mask) for the
    diffusion call where mask == weights. Returns aligned true_coords [B, N, 3].
    """
    mm = lambda a, b, **kw: ttnn.matmul(a, b, compute_kernel_config=compute_kernel_config,
                                        dtype=ttnn.float32, **kw)

    # Work in fp32 throughout: the tensors are tiny ([B,N,3] / [B,3,3]) so the
    # cost is negligible, and the alignment is sensitive enough that bf16 coords
    # measurably degrade the diffusion trajectory.
    true_coords = ttnn.typecast(true_coords, ttnn.float32)
    pred_coords = ttnn.typecast(pred_coords, ttnn.float32)
    weights = ttnn.typecast(weights, ttnn.float32)

    w_sum = ttnn.sum(weights, dim=1, keepdim=True)               # [B,1,1]
    inv_w = ttnn.reciprocal(w_sum)
    true_centroid = ttnn.multiply(ttnn.sum(ttnn.multiply(true_coords, weights), dim=1, keepdim=True), inv_w)
    pred_centroid = ttnn.multiply(ttnn.sum(ttnn.multiply(pred_coords, weights), dim=1, keepdim=True), inv_w)
    t = ttnn.subtract(true_coords, true_centroid)                # [B,N,3]
    p = ttnn.subtract(pred_coords, pred_centroid)

    # cov = (w * p)^T @ t  -> [B,3,3]
    wp = ttnn.multiply(p, weights)
    cov = mm(wp, t, transpose_a=True)                            # [B,3,3]

    # Scale by 1/Frobenius so all singular values land in (0, 1) < sqrt(3)
    fro = ttnn.sqrt(ttnn.sum(ttnn.multiply(cov, cov), dim=[1, 2], keepdim=True))
    Y = ttnn.multiply(cov, ttnn.reciprocal(fro))

    # Newton-Schulz: Y <- 1.5 Y - 0.5 Y Y^T Y  (converges to the polar factor U V^T)
    for _ in range(iters):
        yt = mm(Y, Y, transpose_a=True)            # Y^T Y  [B,3,3]
        yyty = mm(Y, yt)                            # Y (Y^T Y)
        Y = ttnn.subtract(ttnn.multiply(Y, 1.5), ttnn.multiply(yyty, 0.5))

    # aligned = t @ Y^T + pred_centroid
    aligned = ttnn.add(mm(t, Y, transpose_b=True), pred_centroid)
    return aligned


class Distogram(Module):
    """ttnn DistogramModule: z = z + z^T (over token dims); Linear(token_z -> bins).

    Mirrors boltz2.DistogramModule.forward (output before the trailing reshape).
    """

    def __init__(self, state_dict: Weights, compute_kernel_config: ttnn.DeviceComputeKernelConfig):
        super().__init__(state_dict, compute_kernel_config)
        self.weight = self.torch_to_tt("distogram.weight")          # [token_z, bins]
        self.bias = self.torch_to_tt("distogram.bias", transform=lambda x: x)

    def __call__(self, z: ttnn.Tensor) -> ttnn.Tensor:
        z = ttnn.add(z, ttnn.permute(z, (0, 2, 1, 3)))
        return ttnn.linear(
            z, self.weight, bias=self.bias,
            compute_kernel_config=self.compute_kernel_config, core_grid=CORE_GRID_MAIN,
        )


class PreTrunkLinears(Module):
    """Linear/broadcast core of the pre-trunk embedding assembly (boltz2.Boltz2.forward).

    s_init = s_init(s_inputs);  z_core = z_init_1(s_inputs)[:,:,None] + z_init_2(s_inputs)[:,None,:]
                                         + token_bonds(token_bond_feat)
    The rel_pos / contact / type_bond contributions are added separately (their
    integer features are built host-side). All weights are bias-free here.
    """

    def __init__(self, state_dict: Weights, compute_kernel_config: ttnn.DeviceComputeKernelConfig):
        super().__init__(state_dict, compute_kernel_config)
        self.s_init_w = self.torch_to_tt("s_init.weight")
        self.z1_w = self.torch_to_tt("z_init_1.weight")
        self.z2_w = self.torch_to_tt("z_init_2.weight")
        self.token_bonds_w = self.torch_to_tt("token_bonds.weight")

    def __call__(self, s_inputs: ttnn.Tensor, token_bond_feat: ttnn.Tensor):
        s_init = self._lin(s_inputs, self.s_init_w)                 # [1,N,384]
        z1 = self._lin(s_inputs, self.z1_w)                         # [1,N,128]
        z2 = self._lin(s_inputs, self.z2_w)
        n = z1.shape[1]
        z1 = ttnn.reshape(z1, (1, n, 1, z1.shape[-1]))
        z2 = ttnn.reshape(z2, (1, 1, n, z2.shape[-1]))
        z = ttnn.add(z1, z2)                                        # broadcast -> [1,N,N,128]
        z = ttnn.add(z, self._lin(token_bond_feat, self.token_bonds_w))
        return s_init, z


class RelPosLinear(Module):
    """ttnn rel_pos projection as EMBEDDING LOOKUPS (no host [N,N,139] one-hot).

    The relative-position features are a concatenation of one-hots
    (d_res[66], d_tok[66], same_entity[1], d_chain[6]) and the projection is
    `onehot @ W^T`, which equals gathering rows of `W^T` — i.e. an embedding
    lookup per bin. So instead of building a [N,N,139] one-hot on host (slow) and
    uploading it (~73MB) and matmul-ing, we upload the small integer bins and do
    4 device embedding lookups summed. Exact (rows of W^T), and removes the
    biggest featurization cost. r_max=32 -> 66 res/tok classes; s_max=2 -> 6 chain.
    """

    def __init__(self, state_dict: Weights, compute_kernel_config: ttnn.DeviceComputeKernelConfig):
        super().__init__(state_dict, compute_kernel_config)
        Wt = self.weights["linear_layer.weight"].t().contiguous()   # [139, token_z]
        zrow = torch.zeros(1, Wt.shape[1], dtype=Wt.dtype)

        def _tab(rows):
            return ttnn.from_torch(rows, layout=ttnn.ROW_MAJOR_LAYOUT,
                                   device=self.device, dtype=ttnn.bfloat16)
        self.t_res = _tab(Wt[0:66])
        self.t_tok = _tab(Wt[66:132])
        # same_entity is a single feature: index 0 -> 0, index 1 -> W^T[132].
        self.t_ent = _tab(torch.cat([zrow, Wt[132:133]], dim=0))    # [2, token_z]
        self.t_chain = _tab(Wt[133:139])

    def __call__(self, d_res, d_tok, same_ent, d_chain) -> ttnn.Tensor:
        rpe = _pairgrid_embedding(d_res, self.t_res)
        rpe = ttnn.add(rpe, _pairgrid_embedding(d_tok, self.t_tok))
        rpe = ttnn.add(rpe, _pairgrid_embedding(same_ent, self.t_ent))
        rpe = ttnn.add(rpe, _pairgrid_embedding(d_chain, self.t_chain))
        return rpe


EMBED_ROW_CHUNK = 32768  # cap embedding rows per op so the L1 circular buffer fits


def _pairgrid_embedding(ids_nn: ttnn.Tensor, table: ttnn.Tensor) -> ttnn.Tensor:
    """Embed a [1, N, N] integer pair grid -> [1, N, N, d]. Embed as [N, N]
    (batch=N, seq=N) and chunk the batch so the L1 circular buffer fits at large
    N; avoids the [1, N*N] mega-reshape that overflows. Fixed chunks -> trace-safe."""
    _, n, m = ids_nn.shape
    x = ttnn.reshape(ids_nn, (n, m))                  # [N, N], cheap (drop leading 1)
    rows = max(1, EMBED_ROW_CHUNK // m)               # batch rows per embed
    if n <= rows:
        out = ttnn.embedding(x, table, layout=ttnn.TILE_LAYOUT)
    else:
        outs = [ttnn.embedding(ttnn.slice(x, [s, 0], [min(s + rows, n), m]), table,
                               layout=ttnn.TILE_LAYOUT)
                for s in range(0, n, rows)]
        out = ttnn.concat(outs, dim=0)
    return ttnn.reshape(out, (1, n, m, out.shape[-1]))


class TypeBondEmbedding(Module):
    """ttnn token_bonds_type: nn.Embedding(num_bond_types+1, token_z) lookup."""

    def __init__(self, state_dict: Weights, compute_kernel_config: ttnn.DeviceComputeKernelConfig):
        super().__init__(state_dict, compute_kernel_config)
        # ttnn.embedding wants the table in ROW_MAJOR
        self.table = ttnn.from_torch(self.weights["weight"], layout=ttnn.ROW_MAJOR_LAYOUT,
                                     device=self.device, dtype=ttnn.bfloat16)

    def __call__(self, type_bonds_ids: ttnn.Tensor) -> ttnn.Tensor:
        # type_bonds_ids: [1, N, N] uint32 (ROW_MAJOR). ttnn.embedding wants 2D
        # [batch, seq]; flatten the pair grid, embed, restore [1,N,N,d]. Chunk the
        # rows so the embedding circular buffer stays within L1 at large N (512^2).
        return _pairgrid_embedding(type_bonds_ids, self.table)


class ContactConditioning(Module):
    """ttnn ContactConditioning (boltz2.ContactConditioning).

    Host prepares the per-pair contact features (slices of feats + normalized
    threshold); device does the Fourier embedding, the encoder linear, and the
    masked combine with the learned unspecified/unselected encodings.
    """

    def __init__(self, state_dict: Weights, compute_kernel_config: ttnn.DeviceComputeKernelConfig):
        super().__init__(state_dict, compute_kernel_config)
        self.proj_w = self.torch_to_tt("fourier_embedding.proj.weight")      # [1, dim]
        self.proj_b = self.torch_to_tt("fourier_embedding.proj.bias", transform=lambda x: x)
        self.enc_w = self.torch_to_tt("encoder.weight")
        self.enc_b = self.torch_to_tt("encoder.bias", transform=lambda x: x)
        self.enc_unspecified = self.torch_to_tt("encoding_unspecified", transform=lambda x: x)
        self.enc_unselected = self.torch_to_tt("encoding_unselected", transform=lambda x: x)

    def __call__(self, cc_rest, thr_norm, cc0, cc1):
        # cc_rest [1,N,N,3], thr_norm [1,N,N,1], cc0/cc1 [1,N,N,1] (host-built)
        kc = self.compute_kernel_config
        _, n, m, _ = cc_rest.shape
        t = ttnn.reshape(thr_norm, (n * m, 1))
        fourier = ttnn.linear(t, self.proj_w, bias=self.proj_b, compute_kernel_config=kc, core_grid=CORE_GRID_MAIN)
        fourier = ttnn.cos(ttnn.multiply(fourier, 2 * pi))
        fourier = ttnn.reshape(fourier, (1, n, m, fourier.shape[-1]))
        feat = ttnn.concat([cc_rest, thr_norm, fourier], dim=-1)             # [1,N,N,132]
        enc = ttnn.linear(feat, self.enc_w, bias=self.enc_b, compute_kernel_config=kc, core_grid=CORE_GRID_MAIN)
        gate = ttnn.subtract(ttnn.add(ttnn.multiply(cc0, 0.0), 1.0), ttnn.add(cc0, cc1))
        out = ttnn.multiply(enc, gate)
        out = ttnn.add(out, ttnn.multiply(cc0, self.enc_unspecified))
        out = ttnn.add(out, ttnn.multiply(cc1, self.enc_unselected))
        return out


class AtomPairEmbed(Module):
    """AtomEncoder pair-feature embedding terms:
        p = embed_ref_pos(d)*v + embed_ref_dist(d_norm)*v + embed_mask(v)*v
    d/d_norm/v are the weight-free windowed geometry (built host-side, flattened
    to [M, *]); these are the LinearNoBias projections + masking. Returns [M, atom_z].
    """

    def __init__(self, state_dict: Weights, compute_kernel_config: ttnn.DeviceComputeKernelConfig):
        super().__init__(state_dict, compute_kernel_config)
        self.w_pos = self.torch_to_tt("embed_atompair_ref_pos.weight")
        self.w_dist = self.torch_to_tt("embed_atompair_ref_dist.weight")
        self.w_mask = self.torch_to_tt("embed_atompair_mask.weight")

    def __call__(self, d, d_norm, v):
        p = ttnn.multiply(self._lin(d, self.w_pos), v)
        p = ttnn.add(p, ttnn.multiply(self._lin(d_norm, self.w_dist), v))
        p = ttnn.add(p, ttnn.multiply(self._lin(v, self.w_mask), v))
        return p


class AtomEncoder(Module):
    """ttnn AtomEncoder (boltz2.AtomEncoder, structure_prediction=False path).

    Produces the per-atom single rep (q == c) and the windowed atom-pair rep p
    [B*K, W, H, atom_z]. The weight-free windowed geometry (d/d_norm/v) and the
    `to_keys` indexing matrix are built host-side; all projections/MLP run on device.
    """

    def __init__(self, state_dict: Weights, compute_kernel_config: ttnn.DeviceComputeKernelConfig):
        super().__init__(state_dict, compute_kernel_config)
        self.feat_w = self.torch_to_tt("embed_atom_features.weight")
        self.feat_b = self.torch_to_tt("embed_atom_features.bias", transform=lambda x: x)
        self.pair = AtomPairEmbed(state_dict, compute_kernel_config)
        self.cq_w = self.torch_to_tt("c_to_p_trans_q.1.weight")
        self.ck_w = self.torch_to_tt("c_to_p_trans_k.1.weight")
        self.mlp1 = self.torch_to_tt("p_mlp.1.weight")
        self.mlp2 = self.torch_to_tt("p_mlp.3.weight")
        self.mlp3 = self.torch_to_tt("p_mlp.5.weight")
        # structure_prediction branch (present only in diffusion_conditioning scope)
        try:
            self.s2c_norm_w = self.torch_to_tt("s_to_c_trans.0.weight", transform=lambda x: x)
            self.s2c_norm_b = self.torch_to_tt("s_to_c_trans.0.bias", transform=lambda x: x)
            self.s2c_lin_w = self.torch_to_tt("s_to_c_trans.1.weight")
            self.z2p_norm_w = self.torch_to_tt("z_to_p_trans.0.weight", transform=lambda x: x)
            self.z2p_norm_b = self.torch_to_tt("z_to_p_trans.0.bias", transform=lambda x: x)
            self.z2p_lin_w = self.torch_to_tt("z_to_p_trans.1.weight")
            self.structure_prediction = True
        except Exception:
            self.structure_prediction = False

    def __call__(self, atom_feats, d, d_norm, v, idx_T, dims,
                 s_trunk=None, z=None, att=None, att_q=None, att_k=None):
        B, _, K, W, H = dims
        kc = self.compute_kernel_config
        atom_s = self.feat_w.shape[-1]
        c = self._lin(atom_feats, self.feat_w, bias=self.feat_b)        # [1,N,atom_s]
        q = c

        if self.structure_prediction:
            # s_to_c: bmm(atom_to_token, LN+Linear(s_trunk)); c = c + s_to_c
            s2c = ttnn.layer_norm(s_trunk, weight=self.s2c_norm_w, bias=self.s2c_norm_b,
                                  epsilon=1e-5, compute_kernel_config=kc)
            s2c = self._lin(s2c, self.s2c_lin_w)                        # [1,N_tok,atom_s]
            s2c = ttnn.matmul(att, s2c, compute_kernel_config=kc, core_grid=CORE_GRID_MAIN)
            c = ttnn.add(c, s2c)                                        # [1,N,atom_s]

        # windowed pair embedding terms -> [B*K, W, H, atom_z]
        p = self.pair(d, d_norm, v)                                     # [M, atom_z]
        az = p.shape[-1]
        p = ttnn.reshape(p, (B * K, W, H, az))

        if self.structure_prediction:
            # z_to_p: token-pair z -> windowed atom-pair via "bijd,bwki,bwlj->bwkld"
            Nt = z.shape[1]
            zp = ttnn.layer_norm(z, weight=self.z2p_norm_w, bias=self.z2p_norm_b,
                                 epsilon=1e-5, compute_kernel_config=kc)
            zp = self._lin(zp, self.z2p_lin_w)                          # [1,Nt,Nt,az]
            zp = ttnn.reshape(zp, (Nt, Nt * az))
            # step1: t[k,w,j,d] = sum_i att_q[k,w,i] zp[i,j,d]
            t = ttnn.matmul(ttnn.reshape(att_q, (K * W, Nt)), zp, compute_kernel_config=kc,
                            core_grid=CORE_GRID_MAIN)                   # [K*W, Nt*az]
            t = ttnn.reshape(t, (K, W, Nt, az))
            t = ttnn.permute(t, (0, 2, 1, 3))                          # [K, Nt, W, az]
            t = ttnn.reshape(t, (K, Nt, W * az))
            # step2: out[k,w,l,d] = sum_j t[k,w,j,d] att_k[k,l,j]
            ztp = ttnn.matmul(att_k, t, compute_kernel_config=kc)      # [K, H, W*az]
            ztp = ttnn.reshape(ztp, (K, H, W, az))
            ztp = ttnn.permute(ztp, (0, 2, 1, 3))                      # [K, W, H, az]
            p = ttnn.add(p, ztp)

        # + c_to_p_trans_q(c) broadcast over H  ([B*K,W,1,az])
        cq = self._lin(ttnn.relu(c), self.cq_w)                         # [1,N,az]
        cq = ttnn.reshape(cq, (B * K, W, 1, az))
        p = ttnn.add(p, cq)

        # + c_to_p_trans_k(to_keys(c)) broadcast over W  ([B*K,1,H,az])
        c_resh = ttnn.reshape(c, (2 * K, (W // 2) * atom_s))
        ck_in = ttnn.matmul(idx_T, c_resh, compute_kernel_config=self.compute_kernel_config,
                            core_grid=CORE_GRID_MAIN)                   # [hK, (W//2)*atom_s]
        ck_in = ttnn.reshape(ck_in, (B * K, H, atom_s))
        ck = self._lin(ttnn.relu(ck_in), self.ck_w)                     # [B*K,H,az]
        ck = ttnn.reshape(ck, (B * K, 1, H, az))
        p = ttnn.add(p, ck)

        # + p_mlp(p)
        m = self._lin(ttnn.relu(p), self.mlp1)
        m = self._lin(ttnn.relu(m), self.mlp2)
        m = self._lin(ttnn.relu(m), self.mlp3)
        p = ttnn.add(p, m)
        return q, c, p


class AtomEncProjZ(Module):
    """input_embedder.atom_enc_proj_z: LayerNorm(atom_z) + LinearNoBias(atom_z ->
    depth*heads). Projects the atom-pair rep p into the per-layer attention bias."""

    def __init__(self, state_dict: Weights, compute_kernel_config: ttnn.DeviceComputeKernelConfig):
        super().__init__(state_dict, compute_kernel_config)
        self.norm_w = self.torch_to_tt("0.weight", transform=lambda x: x)
        self.norm_b = self.torch_to_tt("0.bias", transform=lambda x: x)
        self.lin_w = self.torch_to_tt("1.weight")

    def __call__(self, p: ttnn.Tensor) -> ttnn.Tensor:
        p = ttnn.layer_norm(p, weight=self.norm_w, bias=self.norm_b, epsilon=1e-5,
                            compute_kernel_config=self.compute_kernel_config)
        return ttnn.linear(p, self.lin_w, compute_kernel_config=self.compute_kernel_config,
                           core_grid=CORE_GRID_MAIN)


class AtomAttentionEncoder(Module):
    """ttnn AtomAttentionEncoder (input_embedder side, structure_prediction=False).

    Reuses the ttnn DiffusionTransformer (atom_level) for the atom transformer,
    replicating the diffusion's windowed bias/mask/keys prep, then aggregates atoms
    to tokens. Inputs c [1,N,atom_s], atom_enc_bias [1,K,W,H,depth*heads], and the
    host tensors atom_mask [1,N], keys_indexing (idx, bf4) and atom_to_token_mean_T
    [1,n_tokens,N]. Returns a [1,n_tokens,token_s].
    """

    def __init__(self, state_dict, compute_kernel_config, n_layers, n_heads, atom_s):
        super().__init__(state_dict, compute_kernel_config)
        self.transformer = DiffusionTransformer(
            n_layers, atom_s, n_heads, True,
            self.scope("atom_encoder.diffusion_transformer"), compute_kernel_config)
        self.a2t_w = self.torch_to_tt("atom_to_token_trans.0.weight")

    def __call__(self, c, atom_enc_bias, atom_mask, keys_indexing, atom_to_token_mean_T, dims):
        B, N, K, W, H = dims
        atom_s = c.shape[-1]
        nlh = atom_enc_bias.shape[-1]
        kc = self.compute_kernel_config

        # windowed attention bias [B*K, depth*heads, W, H] + key mask + sqrt(W) scale
        bias = ttnn.permute(ttnn.reshape(atom_enc_bias, (B * K, W, H, nlh)), (0, 3, 1, 2))
        mask = ttnn.reshape(atom_mask, (2 * K, W // 2, 1))
        mask = ttnn.permute(mask, (1, 2, 0))
        mask = ttnn.matmul(mask, keys_indexing, compute_kernel_config=kc, core_grid=CORE_GRID_MAIN)
        mask = ttnn.permute(mask, (2, 0, 1))
        mask = ttnn.reshape(mask, (K, 1, 1, H))
        mask = ttnn.multiply(ttnn.add(ttnn.multiply(mask, -1.0), 1.0), -1e9)
        bias = ttnn.multiply(ttnn.add(bias, mask), W ** 0.5)

        a = ttnn.reshape(c, (B, K, W, atom_s))
        a = self.transformer(a, a, bias, keys_indexing)            # [B,K,W,atom_s]
        a = ttnn.reshape(a, (B, N, atom_s))

        qa = ttnn.linear(a, self.a2t_w, activation="relu", compute_kernel_config=kc, core_grid=CORE_GRID_MAIN)
        return ttnn.matmul(atom_to_token_mean_T, qa, compute_kernel_config=kc, core_grid=CORE_GRID_MAIN)


class InputEmbedder(Module):
    """ttnn InputEmbedder (boltz2.InputEmbedder, affinity=False) -> s_inputs.

    Composes AtomEncoder + atom_enc_proj_z + AtomAttentionEncoder, then adds the
    res_type / msa_profile linear encodings and the method/modified/cyclic/mol_type
    conditioning embeddings. Integer features (atom feats, windowed d/d_norm/v,
    indexing matrices, token-level ids, atom_to_token) are host-built and fed in.
    """

    def __init__(self, state_dict, compute_kernel_config, n_layers=3, n_heads=4, atom_s=128):
        super().__init__(state_dict, compute_kernel_config)
        self.atom_encoder = AtomEncoder(self.scope("atom_encoder"), compute_kernel_config)
        self.proj_z = AtomEncProjZ(self.scope("atom_enc_proj_z"), compute_kernel_config)
        self.aae = AtomAttentionEncoder(self.scope("atom_attention_encoder"),
                                        compute_kernel_config, n_layers, n_heads, atom_s)
        self.res_type_w = self.torch_to_tt("res_type_encoding.weight")
        self.msa_profile_w = self.torch_to_tt("msa_profile_encoding.weight")
        self.cyclic_w = self.torch_to_tt("cyclic_conditioning_init.weight")

        def emb_table(name):
            return ttnn.from_torch(self.weights[name], layout=ttnn.ROW_MAJOR_LAYOUT,
                                   device=self.device, dtype=ttnn.bfloat16)
        self.method_table = emb_table("method_conditioning_init.weight")
        self.modified_table = emb_table("modified_conditioning_init.weight")
        self.mol_type_table = emb_table("mol_type_conditioning_init.weight")

    def _emb(self, ids, table):
        return ttnn.embedding(ids, table, layout=ttnn.TILE_LAYOUT)

    def __call__(self, h, dims):
        # h: dict of host-built device tensors (see verify script for keys)
        _, c, p = self.atom_encoder(h["atom_feats"], h["d"], h["d_norm"], h["v"], h["idx_T"], dims)
        atom_enc_bias = self.proj_z(p)
        a = self.aae(c, atom_enc_bias, h["atom_mask"], h["keys_idx"], h["att_mean_T"], dims)

        s = ttnn.add(a, self._lin(h["res_type"], self.res_type_w))
        s = ttnn.add(s, self._lin(h["profile_del"], self.msa_profile_w))
        s = ttnn.add(s, self._emb(h["method"], self.method_table))
        s = ttnn.add(s, self._emb(h["modified"], self.modified_table))
        s = ttnn.add(s, self._lin(h["cyclic"], self.cyclic_w))
        s = ttnn.add(s, self._emb(h["mol_type"], self.mol_type_table))
        return s


class PairwiseConditioning(Module):
    """ttnn PairwiseConditioning (boltz2): LN+LinearNoBias over cat(z_trunk, rel_pos)
    then num_transitions residual ttnn Transition blocks. Returns conditioned z."""

    def __init__(self, state_dict, compute_kernel_config, num_transitions=2):
        super().__init__(state_dict, compute_kernel_config)
        self.norm_w = self.torch_to_tt("dim_pairwise_init_proj.0.weight", transform=lambda x: x)
        self.norm_b = self.torch_to_tt("dim_pairwise_init_proj.0.bias", transform=lambda x: x)
        self.proj_w = self.torch_to_tt("dim_pairwise_init_proj.1.weight")
        self.transitions = [
            Transition(self.scope(f"transitions.{i}"), compute_kernel_config)
            for i in range(num_transitions)
        ]

    def __call__(self, z_trunk: ttnn.Tensor, rel_pos: ttnn.Tensor) -> ttnn.Tensor:
        z = ttnn.concat([z_trunk, rel_pos], dim=-1)
        z = ttnn.layer_norm(z, weight=self.norm_w, bias=self.norm_b, epsilon=1e-5,
                            compute_kernel_config=self.compute_kernel_config)
        z = ttnn.linear(z, self.proj_w, compute_kernel_config=self.compute_kernel_config,
                        core_grid=CORE_GRID_MAIN)
        for tr in self.transitions:
            z = ttnn.add(z, tr(z))
        return z


class _StackedProj:
    """A stack of Sequential(LayerNorm(dim), LinearNoBias(dim->heads)); applies each
    to x and concatenates on the last dim (the atom_enc/dec/token bias projections)."""

    def __init__(self, scope_fn, n_layers, kc):
        self.kc = kc
        self.layers = []
        for i in range(n_layers):
            w = scope_fn(f"{i}")
            self.layers.append((
                w.torch_to_tt("0.weight", transform=lambda x: x),
                w.torch_to_tt("0.bias", transform=lambda x: x),
                w.torch_to_tt("1.weight"),
            ))

    def __call__(self, x):
        outs = []
        for nw, nb, lw in self.layers:
            xn = ttnn.layer_norm(x, weight=nw, bias=nb, epsilon=1e-5, compute_kernel_config=self.kc)
            outs.append(ttnn.linear(xn, lw, compute_kernel_config=self.kc, core_grid=CORE_GRID_MAIN))
        return ttnn.concat(outs, dim=-1)


class _ProjScope:
    """Tiny adapter so _StackedProj can build a Module-like with torch_to_tt per layer."""
    def __init__(self, weights, kc):
        self._m = Module(weights, kc)
    def __call__(self, sub):
        inner = Module(self._m.weights.child(sub), self._m.compute_kernel_config)
        return inner


class DiffusionConditioning(Module):
    """ttnn DiffusionConditioning (boltz2): pairwise_conditioner + sp AtomEncoder +
    atom_enc/atom_dec/token_trans bias projections. Returns q, c, atom_enc_bias,
    atom_dec_bias, token_trans_bias (to_keys/windowing handled by caller)."""

    def __init__(self, state_dict, compute_kernel_config,
                 enc_layers=3, dec_layers=3, token_layers=24):
        super().__init__(state_dict, compute_kernel_config)
        kc = compute_kernel_config
        self.pairwise = PairwiseConditioning(self.scope("pairwise_conditioner"), kc)
        self.atom_encoder = AtomEncoder(self.scope("atom_encoder"), kc)
        self.enc_proj = _StackedProj(_ProjScope(self.scope("atom_enc_proj_z"), kc), enc_layers, kc)
        self.dec_proj = _StackedProj(_ProjScope(self.scope("atom_dec_proj_z"), kc), dec_layers, kc)
        self.token_proj = _StackedProj(_ProjScope(self.scope("token_trans_proj_z"), kc), token_layers, kc)

    def __call__(self, s_trunk, z_trunk, rel_pos, atom_host, dims):
        z = self.pairwise(z_trunk, rel_pos)
        q, c, p = self.atom_encoder(
            atom_host["atom_feats"], atom_host["d"], atom_host["d_norm"], atom_host["v"],
            atom_host["idx_T"], dims, s_trunk=s_trunk, z=z,
            att=atom_host["att"], att_q=atom_host["att_q"], att_k=atom_host["att_k"])
        atom_enc_bias = self.enc_proj(p)
        atom_dec_bias = self.dec_proj(p)
        token_trans_bias = self.token_proj(z)
        return q, c, atom_enc_bias, atom_dec_bias, token_trans_bias


class ConfidenceModule(Module):
    """ttnn ConfidenceModule (boltz2, use_separate_heads, all input branches on).

    Reuses RelPosLinear/TypeBondEmbedding/ContactConditioning + the ttnn Pairformer
    (padded like TrunkModule). ``reps`` runs the conditioning + pairformer on device
    to produce s_t/z_t; the torch confidence_heads then computes logits + ptm/iptm
    aggregation (host output-processing). Integer features (rel_pos one-hots, contact
    slices, cdist distogram bins) are host-built and fed in.
    """

    def __init__(self, state_dict, compute_kernel_config, pairformer=None):
        super().__init__(state_dict, compute_kernel_config)
        kc = compute_kernel_config
        L = lambda name: self.torch_to_tt(name)
        LN = lambda name: (self.torch_to_tt(name + ".weight", transform=lambda x: x),
                           self.torch_to_tt(name + ".bias", transform=lambda x: x))
        self.s_inputs_norm = LN("s_inputs_norm")
        self.s_norm = LN("s_norm")
        self.z_norm = LN("z_norm")
        self.s_input_to_s = L("s_input_to_s.weight")
        self.s_to_z = L("s_to_z.weight")
        self.s_to_z_t = L("s_to_z_transpose.weight")
        self.prod_in1 = L("s_to_z_prod_in1.weight")
        self.prod_in2 = L("s_to_z_prod_in2.weight")
        self.prod_out = L("s_to_z_prod_out.weight")
        self.rel_pos = RelPosLinear(self.scope("rel_pos"), kc)
        self.token_bonds_w = L("token_bonds.weight")
        self.type_bonds = TypeBondEmbedding(self.scope("token_bonds_type"), kc)
        self.contact = ContactConditioning(self.scope("contact_conditioning"), kc)
        self.dist_table = ttnn.from_torch(self.weights["dist_bin_pairwise_embed.weight"],
                                          layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
        # The pairformer_stack weights live in a ttnn wrapper (no torch params in
        # the state dict), so reuse the already-loaded inner Pairformer when given.
        self.pairformer = pairformer or Pairformer(8, 32, 4, 24, 16, True, self.scope("pairformer_stack"), kc)

    def _ln(self, x, nb):
        return ttnn.layer_norm(x, weight=nb[0], bias=nb[1], epsilon=1e-5, compute_kernel_config=self.compute_kernel_config)

    def condition(self, s_inputs, s, z, host, dims):
        """Conditioned (s, z) before the pairformer (the confidence-specific logic)."""
        B, N = dims
        si = self._ln(s_inputs, self.s_inputs_norm)
        s = ttnn.add(self._ln(s, self.s_norm), self._lin(si, self.s_input_to_s))
        z = self._ln(z, self.z_norm)
        z = ttnn.add(z, self.rel_pos(host["rel_d_res"], host["rel_d_tok"],
                                     host["rel_same_ent"], host["rel_d_chain"]))
        z = ttnn.add(z, self._lin(host["token_bond_feat"], self.token_bonds_w))
        z = ttnn.add(z, self.type_bonds(host["type_bonds_ids"]))
        z = ttnn.add(z, self.contact(host["cc_rest"], host["thr_norm"], host["cc0"], host["cc1"]))
        sz1 = ttnn.reshape(self._lin(si, self.s_to_z), (B, N, 1, -1))
        sz2 = ttnn.reshape(self._lin(si, self.s_to_z_t), (B, 1, N, -1))
        z = ttnn.add(z, ttnn.add(sz1, sz2))
        p1 = ttnn.reshape(self._lin(si, self.prod_in1), (B, N, 1, -1))
        p2 = ttnn.reshape(self._lin(si, self.prod_in2), (B, 1, N, -1))
        z = ttnn.add(z, self._lin(ttnn.multiply(p1, p2), self.prod_out))
        z = ttnn.add(z, self._dist_embed(host["dist_ids"], N))
        return s, z

    def reps(self, s_inputs, s, z, host, dims):
        """Conditioned (s, z) -> pairformer outputs s_t, z_t (unpadded)."""
        B, N = dims
        s, z = self.condition(s_inputs, s, z, host, dims)
        # Pairformer needs seq padded to PAIRFORMER_PAD_MULTIPLE; pad, run, unpad.
        s_t, z_t = self.pairformer(host["pad_s"](s), host["pad_z"](z),
                                   host["pf_mask"], host["pf_attn"], host["pf_attn"])
        s_t = ttnn.slice(s_t, [0, 0, 0], [B, N, s_t.shape[-1]])
        z_t = ttnn.slice(z_t, [0, 0, 0, 0], [B, N, N, z_t.shape[-1]])
        return s_t, z_t

    def _dist_embed(self, dist_ids, N):
        return _pairgrid_embedding(dist_ids, self.dist_table)
