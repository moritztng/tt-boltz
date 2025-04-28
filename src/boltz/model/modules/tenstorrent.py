import torch, ttnn
from torch import nn
from typing import Tuple, Callable

TRIANGLE_ATTENTION_CHUNK_SIZE = 64
TRANSITION_CHUNK_SIZE = 64
mesh_device = None


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
            device=mesh_device.get_devices()[0],
            dtype=ttnn.float32,
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
        self.in_p = self.torch_to_tt("p_in.weight")
        self.in_g = self.torch_to_tt("g_in.weight")
        self.out_norm_weight = self.torch_to_tt("norm_out.weight")
        self.out_norm_bias = self.torch_to_tt("norm_out.bias")
        self.out_p = self.torch_to_tt("p_out.weight")
        self.out_g = self.torch_to_tt("g_out.weight")

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x_norm_in = ttnn.layer_norm(
            x,
            weight=self.in_norm_weight,
            bias=self.in_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        x = ttnn.multiply(
            ttnn.linear(
                x_norm_in, self.in_p, compute_kernel_config=self.compute_kernel_config
            ),
            ttnn.sigmoid_accurate(
                ttnn.linear(
                    x_norm_in,
                    self.in_g,
                    compute_kernel_config=self.compute_kernel_config,
                )
            ),
        )
        dim = int(x.shape[-1] / 2)
        x = ttnn.permute(
            ttnn.matmul(
                ttnn.permute(
                    x[:, :, :, :dim], (0, 3) + ((2, 1) if self.ending else (1, 2))
                ),
                ttnn.permute(
                    x[:, :, :, dim:], (0, 3) + ((1, 2) if self.ending else (2, 1))
                ),
                compute_kernel_config=self.compute_kernel_config,
            ),
            (0, 2, 3, 1),
        )
        x_norm_out = ttnn.layer_norm(
            x,
            weight=self.out_norm_weight,
            bias=self.out_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        x = ttnn.multiply(
            ttnn.linear(
                x_norm_out, self.out_p, compute_kernel_config=self.compute_kernel_config
            ),
            ttnn.sigmoid_accurate(
                ttnn.linear(
                    x_norm_in,
                    self.out_g,
                    compute_kernel_config=self.compute_kernel_config,
                )
            ),
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
        self.layer_norm_weight = self.torch_to_tt("layer_norm.weight")
        self.layer_norm_bias = self.torch_to_tt("layer_norm.bias")
        self.bias_weight = self.torch_to_tt("linear.weight")
        self.q_weight = self.torch_to_tt("linear_q.weight")
        self.k_weight = self.torch_to_tt("linear_k.weight")
        self.v_weight = self.torch_to_tt("linear_v.weight")
        self.o_weight = self.torch_to_tt("linear_o.weight")
        self.g_weight = self.torch_to_tt("linear_g.weight")

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
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
        )
        triangle_bias = ttnn.reshape(triangle_bias, (1, *triangle_bias.shape))
        triangle_bias = ttnn.permute(triangle_bias, (3, 0, 1, 2))
        two_devices = mesh_device.get_num_devices() == 2
        if two_devices:
            triangle_bias = ttnn.aggregate_as_tensor(
                [
                    ttnn.slice(
                        triangle_bias,
                        [0, 0, 0, 0],
                        [triangle_bias.shape[0] // 2, *tuple(triangle_bias.shape)[1:]],
                    ),
                    ttnn.from_device(
                        ttnn.slice(
                            triangle_bias,
                            [triangle_bias.shape[0] // 2, 0, 0, 0],
                            triangle_bias.shape,
                        )
                    ).to(mesh_device.get_devices()[1]),
                ]
            )

        o_chunks = []
        for chunk_start in range(0, x.shape[0], TRIANGLE_ATTENTION_CHUNK_SIZE):
            x_chunk = x[
                chunk_start : min(
                    chunk_start + TRIANGLE_ATTENTION_CHUNK_SIZE, x.shape[0]
                ),
                :,
                :,
            ]
            q = ttnn.linear(
                x_chunk, self.q_weight, compute_kernel_config=self.compute_kernel_config
            )
            k = ttnn.linear(
                x_chunk, self.k_weight, compute_kernel_config=self.compute_kernel_config
            )
            v = ttnn.linear(
                x_chunk, self.v_weight, compute_kernel_config=self.compute_kernel_config
            )
            q = ttnn.permute(q, (2, 0, 1))
            k = ttnn.permute(k, (2, 0, 1))
            v = ttnn.permute(v, (2, 0, 1))
            q = ttnn.reshape(q, (self.n_heads, self.head_dim, *tuple(q.shape)[1:]))
            k = ttnn.reshape(k, (self.n_heads, self.head_dim, *tuple(k.shape)[1:]))
            v = ttnn.reshape(v, (self.n_heads, self.head_dim, *tuple(v.shape)[1:]))
            q = ttnn.permute(q, (0, 2, 3, 1))
            k = ttnn.permute(k, (0, 2, 1, 3))
            v = ttnn.permute(v, (0, 2, 3, 1))
            if two_devices:
                q = ttnn.aggregate_as_tensor(
                    [
                        ttnn.slice(
                            q,
                            [0, 0, 0, 0],
                            [q.shape[0] // 2, *tuple(q.shape)[1:]],
                        ),
                        ttnn.from_device(
                            ttnn.slice(q, [q.shape[0] // 2, 0, 0, 0], q.shape)
                        ).to(mesh_device.get_devices()[1]),
                    ]
                )
                k = ttnn.aggregate_as_tensor(
                    [
                        ttnn.slice(
                            k,
                            [0, 0, 0, 0],
                            [k.shape[0] // 2, *tuple(k.shape)[1:]],
                        ),
                        ttnn.from_device(
                            ttnn.slice(k, [k.shape[0] // 2, 0, 0, 0], k.shape)
                        ).to(mesh_device.get_devices()[1]),
                    ]
                )
                v = ttnn.aggregate_as_tensor(
                    [
                        ttnn.slice(
                            v,
                            [0, 0, 0, 0],
                            [v.shape[0] // 2, *tuple(v.shape)[1:]],
                        ),
                        ttnn.from_device(
                            ttnn.slice(v, [v.shape[0] // 2, 0, 0, 0], v.shape)
                        ).to(mesh_device.get_devices()[1]),
                    ]
                )
            a = ttnn.matmul(q, k, compute_kernel_config=self.compute_kernel_config)
            a = ttnn.multiply(a, self.head_dim**-0.5)
            a = ttnn.add(a, triangle_bias)
            a = ttnn.softmax(
                a,
                dim=-1,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=self.compute_kernel_config.math_fidelity,
                    math_approx_mode=self.compute_kernel_config.math_approx_mode,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=self.compute_kernel_config.packer_l1_acc,
                ),
                numeric_stable=True,
            )
            o = ttnn.matmul(a, v, compute_kernel_config=self.compute_kernel_config)
            if two_devices:
                o = ttnn.all_gather(o, dim=0)
                o = ttnn.get_device_tensors(o)[0]
            o_chunks.append(o)
        o = ttnn.concat(o_chunks, dim=1)
        o = ttnn.permute(o, (0, 3, 1, 2))
        o = ttnn.reshape(o, (-1, *tuple(o.shape)[2:]))
        o = ttnn.permute(o, (1, 2, 0))
        g = ttnn.linear(
            x, self.g_weight, compute_kernel_config=self.compute_kernel_config
        )
        g = ttnn.sigmoid_accurate(g)
        o = ttnn.multiply(o, g)
        x = ttnn.linear(
            o, self.o_weight, compute_kernel_config=self.compute_kernel_config
        )
        if self.ending:
            x = ttnn.permute(x, (1, 0, 2))
        x = ttnn.reshape(x, (1, *x.shape))
        return x


class AttentionPairBias(Module):
    def __init__(
        self,
        head_dim: int,
        n_heads: int,
        diffusion: bool,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.diffusion = diffusion
        if not diffusion:
            self.norm_s_weight = self.torch_to_tt("norm_s.weight")
            self.norm_s_bias = self.torch_to_tt("norm_s.bias")
        self.q_weight = self.torch_to_tt("proj_q.weight")
        self.q_bias = self.torch_to_tt("proj_q.bias")
        self.k_weight = self.torch_to_tt("proj_k.weight")
        self.v_weight = self.torch_to_tt("proj_v.weight")
        self.g_weight = self.torch_to_tt("proj_g.weight")
        self.z_norm_weight = self.torch_to_tt("proj_z.0.weight")
        self.z_norm_bias = self.torch_to_tt("proj_z.0.bias")
        self.z_weight = self.torch_to_tt("proj_z.1.weight")
        self.o_weight = self.torch_to_tt("proj_o.weight")

    def __call__(self, s: ttnn.Tensor, z: ttnn.Tensor) -> ttnn.Tensor:
        if not self.diffusion:
            s = ttnn.layer_norm(
                s,
                weight=self.norm_s_weight,
                bias=self.norm_s_bias,
                epsilon=1e-5,
                compute_kernel_config=self.compute_kernel_config,
            )
        q = ttnn.linear(
            s,
            self.q_weight,
            bias=self.q_bias,
            compute_kernel_config=self.compute_kernel_config,
        )
        k = ttnn.linear(
            s, self.k_weight, compute_kernel_config=self.compute_kernel_config
        )
        v = ttnn.linear(
            s, self.v_weight, compute_kernel_config=self.compute_kernel_config
        )
        q = ttnn.permute(q, (2, 0, 1))
        k = ttnn.permute(k, (2, 0, 1))
        v = ttnn.permute(v, (2, 0, 1))
        q = ttnn.reshape(q, (self.n_heads, self.head_dim, *tuple(q.shape)[1:]))
        k = ttnn.reshape(k, (self.n_heads, self.head_dim, *tuple(k.shape)[1:]))
        v = ttnn.reshape(v, (self.n_heads, self.head_dim, *tuple(v.shape)[1:]))
        q = ttnn.permute(q, (0, 2, 3, 1))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 3, 1))
        a = ttnn.matmul(q, k, compute_kernel_config=self.compute_kernel_config)
        a = ttnn.multiply(a, self.head_dim**-0.5)
        z = ttnn.layer_norm(
            z,
            weight=self.z_norm_weight,
            bias=self.z_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        z = ttnn.linear(
            z, self.z_weight, compute_kernel_config=self.compute_kernel_config
        )
        z = ttnn.permute(z, (3, 0, 1, 2))
        a = ttnn.add(a, z)
        a = ttnn.softmax(
            a,
            dim=-1,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=self.compute_kernel_config.math_fidelity,
                math_approx_mode=self.compute_kernel_config.math_approx_mode,
                fp32_dest_acc_en=False,
                packer_l1_acc=self.compute_kernel_config.packer_l1_acc,
            ),
            numeric_stable=True,
        )
        o = ttnn.matmul(a, v, compute_kernel_config=self.compute_kernel_config)
        o = ttnn.permute(o, (0, 3, 1, 2))
        o = ttnn.reshape(o, (-1, *tuple(o.shape)[2:]))
        o = ttnn.permute(o, (1, 2, 0))
        g = ttnn.linear(
            s, self.g_weight, compute_kernel_config=self.compute_kernel_config
        )
        g = ttnn.sigmoid_accurate(g)
        o = ttnn.multiply(o, g)
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
            )
            x_1 = ttnn.linear(
                x_norm,
                self.fc1_weight,
                activation="silu",
                compute_kernel_config=self.compute_kernel_config,
            )
            x_2 = ttnn.linear(
                x_norm,
                self.fc2_weight,
                compute_kernel_config=self.compute_kernel_config,
            )
            x = ttnn.multiply(x_1, x_2)
            x = ttnn.linear(
                x,
                self.fc3_weight,
                compute_kernel_config=self.compute_kernel_config,
            )
            return x

        if len(x.shape) < 4:
            x = f(x)
        else:
            x_chunks = []
            for chunk_start in range(0, x.shape[1], TRANSITION_CHUNK_SIZE):
                x_chunk = x[
                    :,
                    chunk_start : min(chunk_start + TRANSITION_CHUNK_SIZE, x.shape[1]),
                    :,
                    :,
                ]
                x_chunk = f(x_chunk)
                x_chunks.append(x_chunk)
            x = ttnn.concat(x_chunks, dim=1)
        return x


class PairformerLayer(Module):
    def __init__(
        self,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        att_head_dim: int,
        att_n_heads: int,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
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
        self.attention_pair_bias = AttentionPairBias(
            att_head_dim,
            att_n_heads,
            False,
            filter_dict(state_dict, "attention"),
            compute_kernel_config,
        )
        self.transition_z = Transition(
            filter_dict(state_dict, "transition_z"), compute_kernel_config
        )
        self.transition_s = Transition(
            filter_dict(state_dict, "transition_s"), compute_kernel_config
        )

    def __call__(
        self, s: ttnn.Tensor, z: ttnn.Tensor
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        z = ttnn.add(
            z,
            self.triangle_multiplication_start(z),
        )
        z = ttnn.add(
            z,
            self.triangle_multiplication_end(z),
        )
        z = ttnn.add(
            z,
            self.triangle_attention_start(z),
        )
        z = ttnn.add(
            z,
            self.triangle_attention_end(z),
        )
        z = ttnn.add(z, self.transition_z(z))
        s = ttnn.add(
            s,
            self.attention_pair_bias(s, z),
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
                filter_dict(state_dict, f"layers.{i}"),
                compute_kernel_config,
            )
            for i in range(n_blocks)
        ]

    def __call__(
        self, s: ttnn.Tensor, z: ttnn.Tensor
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        for block in self.blocks:
            s, z = block(s, z)
        return s, z


class AdaLN(Module):
    def __init__(
        self,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.s_norm_weight = self.torch_to_tt("s_norm.weight")
        self.s_scale_weight = self.torch_to_tt("s_scale.weight")
        self.s_scale_bias = self.torch_to_tt("s_scale.bias")
        self.s_bias_weight = self.torch_to_tt("s_bias.weight")

    def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor) -> ttnn.Tensor:
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
        )
        s_scale = ttnn.sigmoid_accurate(s_scale)
        s_bias = ttnn.linear(
            s, self.s_bias_weight, compute_kernel_config=self.compute_kernel_config
        )
        a = ttnn.multiply(a, s_scale)
        a = ttnn.add(a, s_bias)
        return a


class ConditionedTransitionBlock(Module):
    def __init__(
        self,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.adaln = AdaLN(filter_dict(state_dict, "adaln"), compute_kernel_config)
        self.swish_weight = self.torch_to_tt("swish_gate.0.weight")
        self.a_to_b_weight = self.torch_to_tt("a_to_b.weight")
        self.b_to_a_weight = self.torch_to_tt("b_to_a.weight")
        self.output_projection_weight = self.torch_to_tt("output_projection.0.weight")
        self.output_projection_bias = self.torch_to_tt("output_projection.0.bias")

    def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor) -> ttnn.Tensor:
        a = self.adaln(a, s)
        a_swish = ttnn.linear(
            a, self.swish_weight, compute_kernel_config=self.compute_kernel_config
        )
        dim = int(a_swish.shape[-1] / 2)
        a_swish, gates = a_swish[:, :, :dim], a_swish[:, :, dim:]
        gates = ttnn.silu(gates)
        a_swish = ttnn.multiply(gates, a_swish)
        a_b = ttnn.linear(
            a, self.a_to_b_weight, compute_kernel_config=self.compute_kernel_config
        )
        b = ttnn.multiply(a_swish, a_b)
        s = ttnn.linear(
            s,
            self.output_projection_weight,
            bias=self.output_projection_bias,
            compute_kernel_config=self.compute_kernel_config,
        )
        s = ttnn.sigmoid_accurate(s)
        b_a = ttnn.linear(
            b, self.b_to_a_weight, compute_kernel_config=self.compute_kernel_config
        )
        a = ttnn.multiply(s, b_a)
        return a


class DiffusionTransformerLayer(Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.adaln = AdaLN(filter_dict(state_dict, "adaln"), compute_kernel_config)
        self.attn_pair_bias = AttentionPairBias(
            head_dim=dim // n_heads,
            n_heads=n_heads,
            diffusion=True,
            state_dict=filter_dict(state_dict, "pair_bias_attn"),
            compute_kernel_config=compute_kernel_config,
        )
        self.output_projection_weight = self.torch_to_tt(
            "output_projection_linear.weight"
        )
        self.output_projection_bias = self.torch_to_tt("output_projection_linear.bias")
        self.transition = ConditionedTransitionBlock(
            filter_dict(state_dict, "transition"),
            compute_kernel_config,
        )

    def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor, z: ttnn.Tensor) -> ttnn.Tensor:
        b = self.adaln(a, s)
        b = self.attn_pair_bias(b, z)
        s_o = ttnn.linear(
            s,
            self.output_projection_weight,
            bias=self.output_projection_bias,
            compute_kernel_config=self.compute_kernel_config,
        )
        s_o = ttnn.sigmoid_accurate(s_o)
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
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.layers = [
            DiffusionTransformerLayer(
                dim,
                n_heads,
                filter_dict(state_dict, f"layers.{i}"),
                compute_kernel_config,
            )
            for i in range(n_layers)
        ]
        self.z = None

    def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor, z: ttnn.Tensor) -> ttnn.Tensor:
        if self.z is None:
            self.z = z
        for layer in self.layers:
            a = layer(a, s, self.z)
        return a


class TorchWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = None
        global mesh_device
        if mesh_device is None:
            # ttnn.device.EnablePersistentKernelCache() # be careful, can lead to bugs when profiling etc.
            n_devices = min(len(ttnn.get_device_ids()), 2)
            mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, n_devices))
            mesh_device.enable_program_cache()
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _from_torch(self, x: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            x,
            device=mesh_device.get_devices()[0],
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.float32,
        )

    def _to_torch(self, x: ttnn.Tensor) -> torch.Tensor:
        return torch.Tensor(ttnn.to_torch(x)).to(torch.float32)

    def __del__(self):
        global mesh_device
        if mesh_device is not None:
            for device in mesh_device.get_devices():
                ttnn.DumpDeviceProfiler(device)
            ttnn.close_mesh_device(mesh_device)
            mesh_device = None

class PairformerModule(TorchWrapper):
    def __init__(
        self,
        n_blocks: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        att_head_dim: int,
        att_n_heads: int,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.tri_att_head_dim = tri_att_head_dim
        self.tri_att_n_heads = tri_att_n_heads
        self.att_head_dim = att_head_dim
        self.att_n_heads = att_n_heads

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
            filter_dict(state_dict, prefix[:-1]),
            self.compute_kernel_config,
        )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor = None,
        pair_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return tuple(
            self._to_torch(x)
            for x in self.module(
                self._from_torch(s),
                self._from_torch(z),
            )
        )


class DiffusionTransformerModule(TorchWrapper):
    def __init__(
        self,
        n_layers: int,
        dim: int,
        n_heads: int,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.dim = dim
        self.n_heads = n_heads

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
        self.module = DiffusionTransformer(
            self.n_layers,
            self.dim,
            self.n_heads,
            filter_dict(state_dict, prefix[:-1]),
            self.compute_kernel_config,
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor = None,
        to_keys=None,
        multiplicity: int = 1,
        model_cache: torch.Tensor = None,
    ) -> torch.Tensor:
        return self._to_torch(
            self.module(
                self._from_torch(a),
                self._from_torch(s),
                self._from_torch(z) if z is not None else None,
            )
        )
