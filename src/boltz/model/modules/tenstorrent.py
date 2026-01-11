import torch, ttnn, atexit
from torch import nn
from typing import Tuple, Callable, Dict
from models.common.utility_functions import is_wormhole_b0, is_blackhole
from math import pi

TRIANGLE_MULT_CHUNK_SIZE = 32
TRANSITION_CHUNK_SIZE = 128
USE_FLOAT32 = False

device = None


def cleanup():
    global device
    if device is not None:
        ttnn.close_device(device)


atexit.register(cleanup)


def filter_dict(state_dict: dict, prefix: str, remove: str = "") -> dict:
    if not prefix:
        return state_dict
    prefix += "."
    return {
        key[len(prefix) :].replace(remove, ""): value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


class Module:
    def __init__(
        self,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        self.state_dict = state_dict
        self.compute_kernel_config = compute_kernel_config

    def torch_to_tt(
        self,
        key: str,
        transform: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.t(),
    ) -> ttnn.Tensor:
        return ttnn.from_torch(
            transform(self.state_dict[key]),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=ttnn.float32 if USE_FLOAT32 else ttnn.bfloat16,
        )


class TriangleMultiplication(Module):
    def __init__(
        self,
        ending: bool,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.ending = ending
        self.in_norm_weight = self.torch_to_tt("norm_in.weight")
        self.in_norm_bias = self.torch_to_tt("norm_in.bias")
        self.out_norm_weight = self.torch_to_tt("norm_out.weight")
        self.out_norm_bias = self.torch_to_tt("norm_out.bias")
        g_in_t, p_in_t = [
            self.state_dict[k].t() for k in ["g_in.weight", "p_in.weight"]
        ]
        chunk_size, n_chunks = (
            TRIANGLE_MULT_CHUNK_SIZE,
            g_in_t.shape[1] // TRIANGLE_MULT_CHUNK_SIZE,
        )
        self.n_g_in_chunks = n_chunks
        self.n_pairs = n_chunks // 2
        self.gp_in_weight_fused_chunks = [
            ttnn.from_torch(
                torch.cat(
                    [
                        g_in_t[:, i * chunk_size : (i + 1) * chunk_size],
                        g_in_t[
                            :,
                            (i + self.n_pairs)
                            * chunk_size : (i + self.n_pairs + 1)
                            * chunk_size,
                        ],
                        p_in_t[:, i * chunk_size : (i + 1) * chunk_size],
                        p_in_t[
                            :,
                            (i + self.n_pairs)
                            * chunk_size : (i + self.n_pairs + 1)
                            * chunk_size,
                        ],
                    ],
                    dim=1,
                ),
                layout=ttnn.TILE_LAYOUT,
                device=device,
                dtype=ttnn.float32 if USE_FLOAT32 else ttnn.bfloat16,
            )
            for i in range(self.n_pairs)
        ]
        self.g_out_weight = self.torch_to_tt("g_out.weight")
        self.out_p_weight = self.torch_to_tt("p_out.weight")

    def _transform_chunk(self, chunk, permute_dims):
        old = chunk
        for op, *args in [
            (ttnn.typecast, ttnn.bfloat16),
            (ttnn.permute, permute_dims),
            (ttnn.typecast, ttnn.bfloat8_b),
            (ttnn.reallocate,),
        ]:
            chunk = op(chunk, *args) if args else op(chunk)
            ttnn.deallocate(old)
            old = chunk
        return chunk

    def __call__(self, x: ttnn.Tensor, mask: ttnn.Tensor = None) -> ttnn.Tensor:
        x_norm_in = ttnn.layer_norm(
            x,
            weight=self.in_norm_weight,
            bias=self.in_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        H = x_norm_in.shape[1]
        seq_len_tiles, core_grid = (H + 31) // 32, (
            (10, 13) if is_blackhole() else (8, 8)
        )
        per_core_M, per_core_N = (seq_len_tiles + core_grid[0] - 1) // core_grid[0], (
            seq_len_tiles + core_grid[1] - 1
        ) // core_grid[1]
        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=core_grid[::-1],
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
        mask_unsqueezed = ttnn.unsqueeze(mask, -1) if mask is not None else None
        for i in range(self.n_pairs):
            gp_in_fused = ttnn.experimental.minimal_matmul(
                x_norm_in,
                self.gp_in_weight_fused_chunks[i],
                memory_config=(
                    ttnn.L1_MEMORY_CONFIG if H <= 700 else ttnn.DRAM_MEMORY_CONFIG
                ),
                dtype=ttnn.bfloat8_b,
                compute_kernel_config=self.compute_kernel_config,
            )
            g_in_a, g_in_b, p_in_a, p_in_b = ttnn.chunk(gp_in_fused, chunks=4, dim=-1)
            if H > 700:
                p_in_a = ttnn.to_memory_config(p_in_a, ttnn.L1_MEMORY_CONFIG)
                p_in_b = ttnn.to_memory_config(p_in_b, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(gp_in_fused)
            a_chunk = ttnn.multiply_(
                p_in_a, g_in_a, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]
            )
            b_chunk = ttnn.multiply_(
                p_in_b, g_in_b, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]
            )
            ttnn.deallocate(g_in_a)
            ttnn.deallocate(g_in_b)
            if mask_unsqueezed:
                a_chunk = ttnn.multiply_(a_chunk, mask_unsqueezed)
                b_chunk = ttnn.multiply_(b_chunk, mask_unsqueezed)

            a_chunk = self._transform_chunk(
                a_chunk, (0, 3) + ((2, 1) if self.ending else (1, 2))
            )
            b_chunk = self._transform_chunk(
                b_chunk, (0, 3) + ((1, 2) if self.ending else (2, 1))
            )
            x_chunk = ttnn.matmul(
                a_chunk,
                b_chunk,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=program_config,
                dtype=ttnn.bfloat16,
            )
            ttnn.deallocate(a_chunk)
            ttnn.deallocate(b_chunk)
            x_chunk = ttnn.permute(
                x_chunk,
                (0, 2, 3, 1),
            )
            x = (
                ttnn.clone(x_chunk, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if i == 0
                else ttnn.concat([x, x_chunk], dim=-1)
            )
            ttnn.deallocate(x_chunk)
        x = ttnn.layer_norm(
            x,
            weight=self.out_norm_weight,
            bias=self.out_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        core_grid_opt = ttnn.CoreGrid(y=10, x=11) if is_blackhole() else None
        p_out = ttnn.linear(
            x,
            self.out_p_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=core_grid_opt,
        )
        g_out = ttnn.linear(
            x_norm_in,
            self.g_out_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=core_grid_opt,
        )
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
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.ending = ending
        self.scale = self.head_dim**0.5
        self.layer_norm_weight = self.torch_to_tt("layer_norm.weight")
        self.layer_norm_bias = self.torch_to_tt("layer_norm.bias")
        self.o_weight = self.torch_to_tt("linear_o.weight")
        self.bias_weight = ttnn.multiply_(self.torch_to_tt("linear.weight"), self.scale)
        self.qkvg_weight = ttnn.from_torch(
            torch.cat(
                [
                    self.state_dict["linear_q.weight"],
                    self.state_dict["linear_k.weight"],
                    self.state_dict["linear_v.weight"],
                    self.state_dict["linear_g.weight"],
                ],
                dim=0,
            ).t(),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=ttnn.float32 if USE_FLOAT32 else ttnn.bfloat8_b,
        )

    def __call__(self, x: ttnn.Tensor, mask: ttnn.Tensor = None) -> ttnn.Tensor:
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
        use_optimized_path = mask is None
        seq_len = x.shape[0]
        if use_optimized_path:
            padding = -seq_len % 256
            x = ttnn.pad(x, [(0, padding), (0, padding), (0, 0)], 0)
        triangle_bias = ttnn.linear(
            x,
            self.bias_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=9, x=12) if is_blackhole() else None,
        )
        qkvg = ttnn.experimental.minimal_matmul(
            input_tensor=x,
            weight_tensor=self.qkvg_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat8_b,
        )
        split_idx = 3 * self.head_dim * self.n_heads
        qkv = qkvg[..., :split_idx]
        g = qkvg[..., split_idx:]
        del qkvg
        if use_optimized_path:
            triangle_bias = ttnn.reshape(triangle_bias, (1, *triangle_bias.shape))
            triangle_bias = ttnn.permute(triangle_bias, (3, 0, 1, 2))
            qkv = ttnn.unsqueeze(qkv, 0)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads_boltz(
                qkv,
                num_heads=self.n_heads,
                num_kv_heads=self.n_heads,
                transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            o = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=triangle_bias,
                is_causal=False,
                scale=self.scale**-1,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(
                        (13, 10) if is_blackhole() else (8, 8)
                    ),
                    exp_approx_mode=False,
                    q_chunk_size=256,
                    k_chunk_size=256,
                ),
            )
            o = ttnn.experimental.nlp_concat_heads_boltz(
                o, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            o = ttnn.squeeze(o, 0)
        else:
            if self.ending:
                mask = ttnn.permute(mask, (2, 0, 1))
            else:
                mask = ttnn.permute(mask, (1, 0, 2))
            mask = (mask - 1) * 1e9
            triangle_bias = ttnn.permute(triangle_bias, (2, 0, 1))
            triangle_bias = ttnn.unsqueeze(triangle_bias, 1)
            triangle_bias = ttnn.add(triangle_bias, mask)
            qkv = ttnn.reshape(qkv, (seq_len, seq_len, 3 * self.n_heads, self.head_dim))
            qkv = ttnn.permute(qkv, (2, 0, 1, 3))
            q, k, v = ttnn.chunk(qkv, chunks=3, dim=0)
            a = ttnn.matmul(
                q, k, transpose_b=True, compute_kernel_config=self.compute_kernel_config
            )
            a = ttnn.add_(a, triangle_bias)
            a = ttnn.multiply_(a, self.scale**-1)
            a = ttnn.softmax(
                a,
                dim=-1,
                compute_kernel_config=self.compute_kernel_config,
                numeric_stable=True,
            )
            o = ttnn.matmul(a, v, compute_kernel_config=self.compute_kernel_config)
            o = ttnn.permute(o, (1, 2, 0, 3))
            o = ttnn.reshape(o, (o.shape[0], o.shape[1], -1))
        o = ttnn.multiply_(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        x = ttnn.linear(
            o,
            self.o_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=6, x=12) if is_blackhole() else None,
        )
        if use_optimized_path:
            x = x[:seq_len, :seq_len, :]
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
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.compute_pair_bias = compute_pair_bias
        self.atom_level = atom_level 
        if atom_level:
            self.q_weight = self.torch_to_tt("proj_q.weight")
            self.q_bias = self.torch_to_tt("proj_q.bias")
            self.k_weight = self.torch_to_tt("proj_k.weight")
            self.v_weight = self.torch_to_tt("proj_v.weight")
        else:
            qkv_weight = torch.cat([self.state_dict["proj_q.weight"], self.state_dict["proj_k.weight"], self.state_dict["proj_v.weight"]], dim=0)
            head_dim_padding = -head_dim % 32
            padded_head_dim = head_dim + head_dim_padding
            qkv_weight = qkv_weight.reshape(3 * self.n_heads, head_dim, -1)
            qkv_weight = torch.nn.functional.pad(qkv_weight, (0, 0, 0, head_dim_padding), mode='constant', value=0)
            qkv_weight = qkv_weight.reshape(3 * self.n_heads * padded_head_dim, -1)
            self.qkv_weight = ttnn.from_torch(
                qkv_weight.t(),
                layout=ttnn.TILE_LAYOUT,
                device=device,
                dtype=ttnn.float32 if USE_FLOAT32 else ttnn.bfloat16,
            )
            q_bias = self.state_dict["proj_q.bias"]
            q_bias = q_bias.reshape(self.n_heads, head_dim)
            q_bias = torch.nn.functional.pad(q_bias, (0, head_dim_padding), mode='constant', value=0)
            q_bias = q_bias.reshape(self.n_heads * padded_head_dim)
            qkv_bias = torch.cat([q_bias, torch.zeros(2 * self.n_heads * padded_head_dim, dtype=q_bias.dtype, device=q_bias.device)])
            self.qkv_bias = ttnn.from_torch(
                qkv_bias,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                dtype=ttnn.float32 if USE_FLOAT32 else ttnn.bfloat16,
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
        keys_indexing: ttnn.Tensor = None,
    ) -> ttnn.Tensor:
        if not self.atom_level:
            seq_len = s.shape[1]
            seq_len_padding = -seq_len % 64
            qkv = ttnn.linear(
                ttnn.pad(s, [(0, 0), (0, seq_len_padding), (0, 0)], 0),
                self.qkv_weight,
                bias=self.qkv_bias,
                compute_kernel_config=self.compute_kernel_config,
            )
            qkv = ttnn.unsqueeze(qkv, 0)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv,
                num_heads=self.n_heads,
                num_kv_heads=self.n_heads,
                transpose_k_heads=False,
            )
            q = ttnn.permute(q, (1, 0, 2, 3))
            k = ttnn.permute(k, (1, 0, 2, 3))
            v = ttnn.permute(v, (1, 0, 2, 3))
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
                    core_grid=ttnn.CoreGrid(y=8, x=11) if is_blackhole() else None,
                )
                z = ttnn.permute(z, (3, 0, 1, 2))
                z = ttnn.pad(
                    z, [(0, 0), (0, 0), (0, seq_len_padding), (0, seq_len_padding)], 0
                )
            o = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=z,
                is_causal=False,
                scale=self.head_dim**-0.5,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(
                        (13, 10) if is_blackhole() else (8, 8)
                    ),
                    exp_approx_mode=False,
                    q_chunk_size=64,
                    k_chunk_size=64,
                ),
            )
            o = o[:, :, :seq_len, :self.head_dim]
            o = ttnn.permute(o, (0, 3, 1, 2))
            o = ttnn.reshape(o, (-1, *tuple(o.shape)[2:]))
            o = ttnn.permute(o, (1, 2, 0))
        else:
            B, K, W, D = s.shape
            s_kv = ttnn.reshape(s, (B, 2 * K, W // 2, -1))
            s_kv = ttnn.permute(s_kv, (0, 2, 3, 1))
            s_kv = ttnn.matmul(
                s_kv,
                keys_indexing,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
            )
            s_kv = ttnn.permute(s_kv, (0, 3, 1, 2))
            s_kv = ttnn.reshape(s_kv, (B, K, -1, D))
            q = ttnn.linear(
                s,
                self.q_weight,
                bias=self.q_bias,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            k = ttnn.linear(
                s_kv,
                self.k_weight,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            v = ttnn.linear(
                s_kv,
                self.v_weight,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            perm = (3, 0, 1, 2)
            q = ttnn.permute(q, perm)
            k = ttnn.permute(k, perm)
            v = ttnn.permute(v, perm)
            q = ttnn.reshape(q, (self.n_heads, self.head_dim, *tuple(q.shape)[1:]))
            k = ttnn.reshape(k, (self.n_heads, self.head_dim, *tuple(k.shape)[1:]))
            v = ttnn.reshape(v, (self.n_heads, self.head_dim, *tuple(v.shape)[1:]))
            perm_qv = (2, 0, 3, 4, 1)
            perm_k = (2, 0, 3, 1, 4)
            q, v = ttnn.permute(q, perm_qv), ttnn.permute(v, perm_qv)
            k = ttnn.permute(k, perm_k)
            a = ttnn.matmul(
                q,
                k,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            a = ttnn.multiply_(a, self.head_dim**-0.5)
            a = ttnn.add_(a, z)
            a = ttnn.softmax(
                a,
                dim=-1,
                compute_kernel_config=self.compute_kernel_config,
                numeric_stable=True,
            )
            o = ttnn.matmul(
                a,
                v,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(a)
            ttnn.deallocate(v)
            o = ttnn.permute(o, (0, 1, 4, 2, 3))
            o = ttnn.reshape(o, (B, -1, *tuple(o.shape)[3:]))
            o = ttnn.permute(o, (0, 2, 3, 1))
        g = ttnn.linear(
            s,
            self.g_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG if self.atom_level else None,
        )
        o = ttnn.multiply_(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        if self.atom_level:
            ttnn.deallocate(g)
        x = ttnn.linear(
            o, self.o_weight, compute_kernel_config=self.compute_kernel_config
        )
        return x


class Transition(Module):
    def __init__(
        self,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.norm_weight = self.torch_to_tt("norm.weight")
        self.norm_bias = self.torch_to_tt("norm.bias")
        self.fc1_weight = self.torch_to_tt("fc1.weight")
        self.fc2_weight = self.torch_to_tt("fc2.weight")
        self.fc3_weight = self.torch_to_tt("fc3.weight")

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        def f(x):
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
                dtype=ttnn.bfloat8_b,
                core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
            )
            x_2 = ttnn.linear(
                x_norm,
                self.fc2_weight,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
            )
            ttnn.deallocate(x_norm)
            x = ttnn.multiply_(x_1, x_2)
            ttnn.deallocate(x_2)
            x_dram = ttnn.linear(
                x,
                self.fc3_weight,
                compute_kernel_config=self.compute_kernel_config,
                dtype=ttnn.bfloat8_b,
                core_grid=ttnn.CoreGrid(y=8, x=11) if is_blackhole() else None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(x)
            return x_dram
        if len(x.shape) < 4:
            x = f(x)
        else:
            n_chunks = (x.shape[1] + TRANSITION_CHUNK_SIZE - 1) // TRANSITION_CHUNK_SIZE
            chunks = ttnn.chunk(x, chunks=n_chunks, dim=1)
            x_chunks = [f(chunk) for chunk in chunks]
            x = ttnn.concat(x_chunks, dim=1)
        return x


class PairformerLayer(Module):
    def __init__(
        self,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        att_head_dim: int,
        att_n_heads: int,
        transform_s: bool,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.transform_s = transform_s
        self.triangle_multiplication_start = TriangleMultiplication(
            False, filter_dict(state_dict, "tri_mul_out"), compute_kernel_config
        )
        self.triangle_multiplication_end = TriangleMultiplication(
            True, filter_dict(state_dict, "tri_mul_in"), compute_kernel_config
        )
        self.triangle_attention_start = TriangleAttention(
            tri_att_head_dim,
            tri_att_n_heads,
            False,
            filter_dict(state_dict, "tri_att_start", "mha."),
            compute_kernel_config,
        )
        self.triangle_attention_end = TriangleAttention(
            tri_att_head_dim,
            tri_att_n_heads,
            True,
            filter_dict(state_dict, "tri_att_end", "mha."),
            compute_kernel_config,
        )
        self.transition_z = Transition(
            filter_dict(state_dict, "transition_z"), compute_kernel_config
        )
        if transform_s:
            self.pre_norm_s_weight = self.torch_to_tt("pre_norm_s.weight")
            self.pre_norm_s_bias = self.torch_to_tt("pre_norm_s.bias")
            self.attention_pair_bias = AttentionPairBias(
                att_head_dim,
                att_n_heads,
                True,
                False,
                filter_dict(state_dict, "attention"),
                compute_kernel_config,
            )
            self.transition_s = Transition(
                filter_dict(state_dict, "transition_s"), compute_kernel_config
            )

    def __call__(
        self, s: ttnn.Tensor, z: ttnn.Tensor, mask: ttnn.Tensor = None
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        z = ttnn.add(
            z,
            self.triangle_multiplication_start(z, mask),
        )
        z = ttnn.add(
            z,
            self.triangle_multiplication_end(z, mask),
        )
        z = ttnn.add(
            z,
            self.triangle_attention_start(z, mask),
        )
        z = ttnn.add(
            z,
            self.triangle_attention_end(z, mask),
        )
        z = ttnn.add(z, self.transition_z(z))
        if self.transform_s:
            s_norm = ttnn.layer_norm(
                s,
                weight=self.pre_norm_s_weight,
                bias=self.pre_norm_s_bias,
                epsilon=1e-5,
                compute_kernel_config=self.compute_kernel_config,
            )
            s = ttnn.add(
                s,
                self.attention_pair_bias(
                    s_norm,
                    z,
                ),
            )
            s = ttnn.add(s, self.transition_s(s))
        return s, z


class Pairformer(Module):
    def __init__(
        self,
        n_blocks: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        att_head_dim: int,
        att_n_heads: int,
        transform_s: bool,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.blocks = [
            PairformerLayer(
                tri_att_head_dim,
                tri_att_n_heads,
                att_head_dim,
                att_n_heads,
                transform_s,
                filter_dict(state_dict, f"layers.{i}"),
                compute_kernel_config,
            )
            for i in range(n_blocks)
        ]

    def __call__(
        self, s: ttnn.Tensor, z: ttnn.Tensor, mask: ttnn.Tensor = None
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        for block in self.blocks:
            s, z = block(s, z, mask)
        return s, z


class AdaLN(Module):
    def __init__(
        self,
        atom_level: bool,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.atom_level = atom_level
        self.s_norm_weight = self.torch_to_tt("s_norm.weight")
        self.s_scale_weight = self.torch_to_tt("s_scale.weight")
        self.s_scale_bias = self.torch_to_tt("s_scale.bias")
        self.s_bias_weight = self.torch_to_tt("s_bias.weight")

    def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor) -> ttnn.Tensor:
        memory_config = ttnn.L1_MEMORY_CONFIG if self.atom_level else None
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
        )
        s_bias = ttnn.linear(
            s,
            self.s_bias_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
        )
        a = ttnn.multiply_(a, s_scale, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        a = ttnn.add_(a, s_bias)
        return a


class ConditionedTransitionBlock(Module):
    def __init__(
        self,
        atom_level: bool,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.atom_level = atom_level
        self.adaln = AdaLN(
            atom_level, filter_dict(state_dict, "adaln"), compute_kernel_config
        )
        self.swish_weight = self.torch_to_tt("swish_gate.0.weight")
        self.a_to_b_weight = self.torch_to_tt("a_to_b.weight")
        self.b_to_a_weight = self.torch_to_tt("b_to_a.weight")
        self.output_projection_weight = self.torch_to_tt("output_projection.0.weight")
        self.output_projection_bias = self.torch_to_tt("output_projection.0.bias")

    def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor) -> ttnn.Tensor:
        memory_config = ttnn.L1_MEMORY_CONFIG if self.atom_level else None
        a = self.adaln(a, s)
        a_swish = ttnn.linear(
            a,
            self.swish_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
            core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
        )
        a_swish, gates = ttnn.chunk(a_swish, chunks=2, dim=-1)
        a_swish = ttnn.multiply_(gates, a_swish, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        a_b = ttnn.linear(
            a,
            self.a_to_b_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
        )
        b = ttnn.multiply_(a_swish, a_b)
        if self.atom_level:
            ttnn.deallocate(a_b)
        s = ttnn.linear(
            s,
            self.output_projection_weight,
            bias=self.output_projection_bias,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
        )
        b_a = ttnn.linear(
            b,
            self.b_to_a_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
        )
        if self.atom_level:
            ttnn.deallocate(b)
        a = ttnn.multiply_(s, b_a, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        return a


class DiffusionTransformerLayer(Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        atom_level: bool,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.atom_level = atom_level
        self.adaln = AdaLN(
            atom_level, filter_dict(state_dict, "adaln"), compute_kernel_config
        )
        self.attn_pair_bias = AttentionPairBias(
            head_dim=dim // n_heads,
            n_heads=n_heads,
            compute_pair_bias=False,
            atom_level=atom_level,
            state_dict=filter_dict(state_dict, "pair_bias_attn"),
            compute_kernel_config=compute_kernel_config,
        )
        self.output_projection_weight = self.torch_to_tt(
            "output_projection_linear.weight"
        )
        self.output_projection_bias = self.torch_to_tt("output_projection_linear.bias")
        self.transition = ConditionedTransitionBlock(
            atom_level,
            filter_dict(state_dict, "transition"),
            compute_kernel_config,
        )

    def __call__(
        self,
        a: ttnn.Tensor,
        s: ttnn.Tensor,
        z: ttnn.Tensor,
        keys_indexing: ttnn.Tensor,
    ) -> ttnn.Tensor:
        b = self.adaln(a, s)
        if not self.atom_level:
            b = self.attn_pair_bias(b, z)
        else:
            b = self.attn_pair_bias(b, z, keys_indexing)
        if not hasattr(self, "s_o"):
            s_o = ttnn.linear(
                s,
                self.output_projection_weight,
                bias=self.output_projection_bias,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
                activation="sigmoid",
            )
            if self.atom_level:
                self.s_o = s_o
        else:
            s_o = self.s_o
        b = ttnn.multiply(s_o, b)
        a = ttnn.add(a, b)
        a_t = self.transition(a, s)
        a = ttnn.add(a, a_t)
        return a


class DiffusionTransformer(Module):
    def __init__(
        self,
        n_layers: int,
        dim: int,
        n_heads: int,
        atom_level: bool,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.layers = [
            DiffusionTransformerLayer(
                dim,
                n_heads,
                atom_level,
                filter_dict(state_dict, f"layers.{i}"),
                compute_kernel_config,
            )
            for i in range(n_layers)
        ]

    def __call__(
        self,
        a: ttnn.Tensor,
        s: ttnn.Tensor,
        z: ttnn.Tensor,
        keys_indexing: ttnn.Tensor = None,
    ) -> ttnn.Tensor:
        dim = z.shape[0] // len(self.layers)
        for i, layer in enumerate(self.layers):
            a = layer(a, s, z[i * dim : (i + 1) * dim, :, :, :], keys_indexing)
        return a


class PairWeightedAveraging(Module):
    def __init__(
        self,
        head_dim: int,
        n_heads: int,
        state_dict: dict,
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

    def __call__(self, m: ttnn.Tensor, z: ttnn.Tensor) -> ttnn.Tensor:
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
        for i in range(self.n_heads):
            b = ttnn.linear(
                z,
                self.z_weight[:, i : i + 1],
                compute_kernel_config=self.compute_kernel_config,
                core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
            )
            b = ttnn.permute(b, (2, 0, 1))
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
                core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
            )
            v = ttnn.permute(v, (0, 2, 1))
            o = ttnn.matmul(
                v,
                w,
                transpose_b=True,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
            )
            del v, w
            o = ttnn.permute(o, (0, 2, 1))
            g = ttnn.linear(
                m,
                self.g_weight[:, i * self.head_dim : (i + 1) * self.head_dim],
                compute_kernel_config=self.compute_kernel_config,
                core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
            )
            o = ttnn.multiply(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            del g
            o = ttnn.linear(
                o,
                self.o_weight[i * self.head_dim : (i + 1) * self.head_dim, :],
                compute_kernel_config=self.compute_kernel_config,
                core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
            )
            if i == 0:
                o_out = o
            else:
                o_out = ttnn.add(o_out, o)
        o_out = ttnn.reshape(o_out, (1, *o_out.shape))
        return o_out


class OuterProductMean(Module):
    def __init__(
        self,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.norm_weight = self.torch_to_tt("norm.weight")
        self.norm_bias = self.torch_to_tt("norm.bias")
        self.a_weight = self.torch_to_tt("proj_a.weight")
        self.b_weight = self.torch_to_tt("proj_b.weight")
        self.o_weight = self.torch_to_tt("proj_o.weight")
        self.o_bias = self.torch_to_tt("proj_o.bias")

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
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
            core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
        )
        b = ttnn.linear(
            m,
            self.b_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
        )
        S, I, C = a.shape
        _, J, D = b.shape
        a = ttnn.permute(a, (1, 2, 0))
        a = ttnn.reshape(a, (-1, S))
        b = ttnn.permute(b, (2, 1, 0))
        b = ttnn.to_layout(b, ttnn.ROW_MAJOR_LAYOUT)
        b = ttnn.reshape(b, (-1, S))
        b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)
        z = ttnn.matmul(a, b, transpose_b=True, compute_kernel_config=self.compute_kernel_config)
        z = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
        z = ttnn.reshape(z, (I, C * D, J))
        z = ttnn.to_layout(z, ttnn.TILE_LAYOUT)
        z = ttnn.permute(z, (0, 2, 1))
        z = ttnn.multiply(z, 1 / S)
        z = ttnn.linear(
            z,
            self.o_weight,
            bias=self.o_bias,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
        )
        z = ttnn.reshape(z, (1, *z.shape))
        return z


class MSALayer(Module):
    def __init__(
        self,
        avg_head_dim: int,
        avg_n_heads: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.msa_transition = Transition(
            filter_dict(state_dict, "msa_transition"), compute_kernel_config
        )
        self.pair_weighted_averaging = PairWeightedAveraging(
            head_dim=avg_head_dim,
            n_heads=avg_n_heads,
            state_dict=filter_dict(state_dict, "pair_weighted_averaging"),
            compute_kernel_config=compute_kernel_config,
        )
        self.outer_product_mean = OuterProductMean(
            state_dict=filter_dict(state_dict, "outer_product_mean"),
            compute_kernel_config=compute_kernel_config,
        )
        self.pairformer_layer = PairformerLayer(
            tri_att_head_dim,
            tri_att_n_heads,
            None,
            None,
            False,
            filter_dict(state_dict, f"pairformer_layer"),
            compute_kernel_config,
        )

    def __call__(
        self, z: ttnn.Tensor, m: ttnn.Tensor
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        m = ttnn.add(m, self.pair_weighted_averaging(m, z))
        m = ttnn.add(m, self.msa_transition(m))
        z = ttnn.add(z, self.outer_product_mean(m))
        z = self.pairformer_layer(None, z)[1]
        return z, m


class MSA(Module):
    def __init__(
        self,
        n_blocks: int,
        avg_head_dim: int,
        avg_n_heads: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        state_dict: dict,
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
                filter_dict(state_dict, f"layers.{i}"),
                compute_kernel_config,
            )
            for i in range(n_blocks)
        ]

    def __call__(self, z: ttnn.Tensor, m: ttnn.Tensor, emb: ttnn.Tensor) -> ttnn.Tensor:
        m = ttnn.linear(
            m,
            self.msa_weight,
            compute_kernel_config=self.compute_kernel_config,
        )
        m = ttnn.add(
            m,
            ttnn.linear(
                emb,
                self.s_weight,
                compute_kernel_config=self.compute_kernel_config,
            ),
        )
        for block in self.blocks:
            z, m = block(z, m)
        return z


class Diffusion(Module):
    def __init__(
        self,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
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
            filter_dict(state_dict, "single_conditioner.transitions.0"),
            compute_kernel_config,
        )
        self.conditioner_transition_1 = Transition(
            filter_dict(state_dict, "single_conditioner.transitions.1"),
            compute_kernel_config,
        )
        self.a_norm_bias = self.torch_to_tt("a_norm.bias")
        self.r_to_q_weight = self.torch_to_tt(
            "atom_attention_encoder.r_to_q_trans.weight"
        )
        self.encoder = DiffusionTransformer(
            n_layers=3,
            dim=128,
            n_heads=4,
            atom_level=True,
            state_dict=filter_dict(
                state_dict, f"atom_attention_encoder.atom_encoder.diffusion_transformer"
            ),
            compute_kernel_config=compute_kernel_config,
        )
        self.atom_to_token_weight = self.torch_to_tt(
            "atom_attention_encoder.atom_to_token_trans.0.weight"
        )
        self.s_to_a_norm_weight = self.torch_to_tt("s_to_a_linear.0.weight")
        self.s_to_a_norm_bias = self.torch_to_tt("s_to_a_linear.0.bias")
        self.s_to_a_linear_weight = self.torch_to_tt("s_to_a_linear.1.weight")
        self.token_transformer = DiffusionTransformer(
            n_layers=24,
            dim=2 * 384,
            n_heads=16,
            atom_level=False,
            state_dict=filter_dict(state_dict, f"token_transformer"),
            compute_kernel_config=compute_kernel_config,
        )
        self.a_norm_weight = self.torch_to_tt("a_norm.weight")
        self.a_norm_bias = self.torch_to_tt("a_norm.bias")
        self.a_to_q_weight = self.torch_to_tt(
            "atom_attention_decoder.a_to_q_trans.weight"
        )
        self.decoder = DiffusionTransformer(
            n_layers=3,
            dim=128,
            n_heads=4,
            atom_level=True,
            state_dict=filter_dict(
                state_dict, f"atom_attention_decoder.atom_decoder.diffusion_transformer"
            ),
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
    ) -> ttnn.Tensor:
        W = 32
        B, N, D = q.shape
        NW = N // W
        r_to_q = ttnn.linear(
            r,
            self.r_to_q_weight,
            compute_kernel_config=self.compute_kernel_config,
        )
        q = ttnn.add(q, r_to_q)
        q = ttnn.reshape(q, (B, NW, W, -1))
        c = ttnn.reshape(c, (B, NW, W, -1))
        q = self.encoder(q, c, bias_encoder, keys_indexing)
        q = ttnn.reshape(q, (B, NW * W, D))
        a = ttnn.linear(
            q,
            self.atom_to_token_weight,
            compute_kernel_config=self.compute_kernel_config,
            activation="relu",
        )
        a = ttnn.matmul(
            a,
            atom_to_token_normed,
            transpose_a=True,
            compute_kernel_config=self.compute_kernel_config,
        )
        a = ttnn.permute(a, (0, 2, 1))
        s = ttnn.concat([s_trunk, s_inputs], dim=-1)
        s = ttnn.layer_norm(
            s,
            weight=self.conditioner_norm_weight,
            bias=self.conditioner_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        s = ttnn.linear(
            s,
            self.conditioner_embed_weight,
            bias=self.conditioner_embed_bias,
            compute_kernel_config=self.compute_kernel_config,
        )
        times = ttnn.unsqueeze(times, 1)
        fourier = ttnn.linear(
            times,
            self.conditioner_fourier_embed_weight,
            bias=self.conditioner_fourier_embed_bias,
            compute_kernel_config=self.compute_kernel_config,
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
        )
        fourier = ttnn.unsqueeze(fourier, 1)
        s = ttnn.add(s, fourier)
        s = ttnn.add(s, self.conditioner_transition_0(s))
        s = ttnn.add(s, self.conditioner_transition_1(s))
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
        )
        a = ttnn.add(a, s_to_a)
        a = self.token_transformer(a, s, bias_token)
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
        )
        a_to_q = ttnn.permute(a_to_q, (0, 2, 1))
        a_to_q = ttnn.matmul(
            a_to_q,
            atom_to_token,
            transpose_b=True,
            compute_kernel_config=self.compute_kernel_config,
        )
        a_to_q = ttnn.permute(a_to_q, (0, 2, 1))
        q = ttnn.add(q, a_to_q)
        q = ttnn.reshape(q, (B, NW, W, -1))
        c = ttnn.reshape(c, (B, NW, W, -1))
        q = self.decoder(q, c, bias_decoder, keys_indexing)
        q = ttnn.reshape(q, (B, NW * W, D))
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
        )
        return r_update


class TorchWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = None
        global device
        if device is None:
            ttnn.device.EnablePersistentKernelCache()  # be careful, can lead to bugs when profiling etc.
            args = {"device_id": 0}
            if is_wormhole_b0():
                args["dispatch_core_config"] = ttnn.DispatchCoreConfig(
                    ttnn.device.DispatchCoreType.ETH, ttnn.DispatchCoreAxis.ROW
                )
            device = ttnn.open_device(**args)
            device.enable_program_cache()
        self.compute_kernel_config = ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _from_torch(self, x: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            x,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.float32 if USE_FLOAT32 else ttnn.bfloat16,
        )

    def _to_torch(self, x: ttnn.Tensor) -> torch.Tensor:
        return torch.Tensor(ttnn.to_torch(x)).to(torch.float32)


class PairformerModule(TorchWrapper):
    def __init__(
        self,
        n_blocks: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        att_head_dim: int,
        att_n_heads: int,
        transform_s: bool,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.tri_att_head_dim = tri_att_head_dim
        self.tri_att_n_heads = tri_att_n_heads
        self.att_head_dim = att_head_dim
        self.att_n_heads = att_n_heads
        self.transform_s = transform_s

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.module = Pairformer(
            self.n_blocks,
            self.tri_att_head_dim,
            self.tri_att_n_heads,
            self.att_head_dim,
            self.att_n_heads,
            self.transform_s,
            filter_dict(state_dict, prefix[:-1]),
            self.compute_kernel_config,
        )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor = None,
        pair_mask: torch.Tensor = None,
        use_kernels: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return tuple(
            self._to_torch(x) if x is not None else None
            for x in self.module(
                self._from_torch(s) if s is not None else None,
                self._from_torch(z),
                self._from_torch(mask) if (mask is not None) and (s is None) else None,
            )
        )


class DiffusionModule(TorchWrapper):
    def __init__(self):
        super().__init__()
        self.first_forward_pass = True

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.module = Diffusion(
            filter_dict(state_dict, prefix[:-1]),
            self.compute_kernel_config,
        )

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
        W = 32
        H = 128
        B, N, _ = q.shape
        NW = N // W
        K = B * NW
        TOKEN_TRANSFORMER_DIM = 2 * 384
        TOKEN_TRANSFORMER_N_HEADS = 16
        if self.first_forward_pass:
            self.first_forward_pass = False
            self.s_inputs = self._from_torch(s_inputs)
            self.s_trunk = self._from_torch(s_trunk)
            self.q = self._from_torch(q if r.shape[0] == q.shape[0] else torch.repeat_interleave(q, r.shape[0], dim=0))
            self.c = self._from_torch(c if r.shape[0] == c.shape[0] else torch.repeat_interleave(c, r.shape[0], dim=0))

            self.keys_indexing = self._from_torch(keys_indexing)

            mask = self._from_torch(mask)
            mask = ttnn.reshape(mask, (2 * K, W // 2, -1))
            mask = ttnn.permute(mask, (1, 2, 0))
            mask = ttnn.matmul(
                mask,
                self.keys_indexing,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
            )
            mask = ttnn.permute(mask, (2, 0, 1))
            mask = ttnn.reshape(mask, (1, K, 1, -1))
            mask = (-1 * mask + 1) * -1e9

            bias = self._from_torch(bias_encoder)
            bias = ttnn.reshape(bias, (B * NW, W, H, -1))
            bias = ttnn.permute(bias, (3, 0, 1, 2))
            self.bias_encoder = ttnn.add(bias, mask)

            bias = self._from_torch(bias_decoder)
            bias = ttnn.reshape(bias, (B * NW, W, H, -1))
            bias = ttnn.permute(bias, (3, 0, 1, 2))
            self.bias_decoder = ttnn.add(bias, mask)

            bias = self._from_torch(bias_token)
            bias = ttnn.multiply_(
                bias, (TOKEN_TRANSFORMER_DIM / TOKEN_TRANSFORMER_N_HEADS) ** 0.5
            )
            bias = ttnn.permute(bias, (3, 0, 1, 2))
            seq_len_padding = -bias.shape[-1] % 64
            self.bias_token = ttnn.pad(
                bias,
                [(0, 0), (0, 0), (0, seq_len_padding), (0, seq_len_padding)],
                0,
            )

            self.atom_to_token = self._from_torch(atom_to_token)
            self.atom_to_token_normed = ttnn.multiply(
                self.atom_to_token,
                ttnn.reciprocal(
                    ttnn.sum(self.atom_to_token, dim=1, keepdim=True) + 1e-6
                ),
            )
        return self._to_torch(
            self.module(
                self._from_torch(r),
                self._from_torch(times),
                self.s_inputs,
                self.s_trunk,
                self.q,
                self.c,
                self.bias_encoder,
                self.bias_token,
                self.bias_decoder,
                self.keys_indexing,
                self.atom_to_token,
                self.atom_to_token_normed,
            )
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

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.module = MSA(
            self.n_blocks,
            self.avg_head_dim,
            self.avg_n_heads,
            self.tri_att_head_dim,
            self.tri_att_n_heads,
            filter_dict(state_dict, prefix[:-1]),
            self.compute_kernel_config,
        )

    def forward(
        self,
        z: torch.Tensor,
        emb: torch.Tensor,
        feats: Dict[str, torch.Tensor],
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
        return self._to_torch(
            self.module(
                self._from_torch(z),
                self._from_torch(m),
                self._from_torch(emb),
            )
        )
