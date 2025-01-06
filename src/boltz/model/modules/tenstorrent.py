import torch, ttnn
from torch import nn
from typing import Tuple, Callable

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
    def __init__(self, device: ttnn._ttnn.device.Device, state_dict: dict):
        self.device = device
        self.state_dict = state_dict

    def torch_to_tt(
        self,
        key: str,
        transform: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.t(),
    ) -> ttnn.Tensor:
        return ttnn.from_torch(
            transform(self.state_dict[key]),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            dtype=ttnn.bfloat16,
        )


class TriangleMultiplication(Module):
    def __init__(
        self, ending: bool, device: ttnn._ttnn.device.Device, state_dict: dict
    ):
        super().__init__(device, state_dict)
        self.ending = ending
        self.in_norm_weight = self.torch_to_tt(
            "norm_in.weight", lambda x: x.expand(32, -1)
        )
        self.in_norm_bias = self.torch_to_tt("norm_in.bias", lambda x: x.expand(32, -1))
        self.in_p = self.torch_to_tt("p_in.weight")
        self.in_g = self.torch_to_tt("g_in.weight")
        self.out_norm_weight = self.torch_to_tt(
            "norm_out.weight", lambda x: x.expand(32, -1)
        )
        self.out_norm_bias = self.torch_to_tt(
            "norm_out.bias", lambda x: x.expand(32, -1)
        )
        self.out_p = self.torch_to_tt("p_out.weight")
        self.out_g = self.torch_to_tt("g_out.weight")

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x_norm_in = ttnn.layer_norm(
            x, weight=self.in_norm_weight, bias=self.in_norm_bias, epsilon=1e-5
        )
        x_p = ttnn.linear(x_norm_in, self.in_p)
        x_g = ttnn.linear(x_norm_in, self.in_g)
        x_s = ttnn.sigmoid(x_g)
        x = ttnn.multiply(x_p, x_s)
        dim = int(x.shape[-1] / 2)
        x = ttnn.permute(
            ttnn.matmul(
                ttnn.permute(
                    x[:, :, :, :dim], (0, 3) + ((2, 1) if self.ending else (1, 2))
                ),
                ttnn.permute(
                    x[:, :, :, dim:], (0, 3) + ((1, 2) if self.ending else (2, 1))
                ),
            ),
            (0, 2, 3, 1),
        )
        x_norm_out = ttnn.layer_norm(
            x, weight=self.out_norm_weight, bias=self.out_norm_bias, epsilon=1e-5
        )
        x_p = ttnn.linear(x_norm_out, self.out_p)
        x_g = ttnn.linear(x_norm_in, self.out_g)
        x_s = ttnn.sigmoid(x_g)
        x = ttnn.multiply(x_p, x_s)
        return x


class TriangleAttention(Module):
    def __init__(
        self,
        head_dim: int,
        n_heads: int,
        ending: bool,
        device: ttnn._ttnn.device.Device,
        state_dict: dict,
    ):
        super().__init__(device, state_dict)
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.ending = ending
        self.layer_norm_weight = self.torch_to_tt(
            "layer_norm.weight", lambda x: x.expand(32, -1)
        )
        self.layer_norm_bias = self.torch_to_tt(
            "layer_norm.bias", lambda x: x.expand(32, -1)
        )
        self.bias_weight = self.torch_to_tt("linear.weight")
        self.q_weight = self.torch_to_tt("linear_q.weight")
        self.k_weight = self.torch_to_tt("linear_k.weight")
        self.v_weight = self.torch_to_tt("linear_v.weight")
        self.o_weight = self.torch_to_tt("linear_o.weight")
        self.g_weight = self.torch_to_tt("linear_g.weight")

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.reshape(x, tuple(x.shape)[1:])
        if self.ending:
            x = ttnn.permute(x, (1, 0, 2))
        x = ttnn.layer_norm(
            x, weight=self.layer_norm_weight, bias=self.layer_norm_bias, epsilon=1e-5
        )
        triangle_bias = ttnn.linear(x, self.bias_weight)
        triangle_bias = ttnn.permute(triangle_bias, (2, 0, 1))
        triangle_bias = ttnn.reshape(triangle_bias, (1, *triangle_bias.shape))
        q = ttnn.linear(x, self.q_weight)
        k = ttnn.linear(x, self.k_weight)
        v = ttnn.linear(x, self.v_weight)
        q = ttnn.reshape(q, (*tuple(q.shape)[:2], self.n_heads, self.head_dim))
        k = ttnn.reshape(k, (*tuple(k.shape)[:2], self.n_heads, self.head_dim))
        v = ttnn.reshape(v, (*tuple(v.shape)[:2], self.n_heads, self.head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 3, 1))
        v = ttnn.permute(v, (0, 2, 1, 3))
        a = ttnn.matmul(q, k)
        a = ttnn.multiply(a, self.head_dim**-0.5)
        a = ttnn.add(a, triangle_bias)
        a = ttnn.softmax(a, -1)
        o = ttnn.matmul(a, v)
        o = ttnn.permute(o, (0, 2, 1, 3))
        o = ttnn.reshape(o, (*tuple(o.shape)[:2], -1))
        g = ttnn.linear(x, self.g_weight)
        g = ttnn.sigmoid(g)
        o = ttnn.multiply(o, g)
        x = ttnn.linear(o, self.o_weight)
        if self.ending:
            x = ttnn.permute(x, (1, 0, 2))
        x = ttnn.reshape(x, (1, *x.shape))
        return x


class AttentionPairBias(Module):
    def __init__(
        self,
        head_dim: int,
        n_heads: int,
        initial_norm: bool,
        device: ttnn._ttnn.device.Device,
        state_dict: dict,
    ):
        super().__init__(device, state_dict)
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.initial_norm = initial_norm
        if initial_norm:
            self.norm_s_weight = self.torch_to_tt(
                "norm_s.weight", lambda x: x.expand(32, -1)
            )
            self.norm_s_bias = self.torch_to_tt(
                "norm_s.bias", lambda x: x.expand(32, -1)
            )
        self.q_weight = self.torch_to_tt("proj_q.weight")
        self.q_bias = self.torch_to_tt("proj_q.bias")
        self.k_weight = self.torch_to_tt("proj_k.weight")
        self.v_weight = self.torch_to_tt("proj_v.weight")
        self.g_weight = self.torch_to_tt("proj_g.weight")
        self.z_norm_weight = self.torch_to_tt("proj_z.0.weight")
        self.z_norm_bias = self.torch_to_tt("proj_z.0.bias")
        self.z_weight = self.torch_to_tt("proj_z.1.weight")
        self.o_weight = self.torch_to_tt("proj_o.weight")
        self.device = device

    def __call__(self, s: ttnn.Tensor, z: ttnn.Tensor) -> ttnn.Tensor:
        if self.initial_norm:
            s = ttnn.layer_norm(
                s, weight=self.norm_s_weight, bias=self.norm_s_bias, epsilon=1e-5
            )
        q = ttnn.linear(s, self.q_weight, bias=self.q_bias)
        k = ttnn.linear(s, self.k_weight)
        v = ttnn.linear(s, self.v_weight)
        q = ttnn.reshape(q, (*tuple(q.shape)[:2], self.n_heads, self.head_dim))
        k = ttnn.reshape(k, (*tuple(k.shape)[:2], self.n_heads, self.head_dim))
        v = ttnn.reshape(v, (*tuple(v.shape)[:2], self.n_heads, self.head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 3, 1))
        v = ttnn.permute(v, (0, 2, 1, 3))
        a = ttnn.matmul(q, k)
        a = ttnn.multiply(a, self.head_dim**-0.5)
        z = ttnn.layer_norm(
            z, weight=self.z_norm_weight, bias=self.z_norm_bias, epsilon=1e-5
        )
        z = ttnn.linear(z, self.z_weight)
        z = ttnn.permute(z, (0, 3, 1, 2))
        a = ttnn.add(a, z)
        # diffusion transformer second layer precision to low
        a = ttnn.softmax(a, -1)
        o = ttnn.matmul(a, v)
        o = ttnn.permute(o, (0, 2, 1, 3))
        o = ttnn.to_torch(o)
        o = ttnn.from_torch(
            o.reshape(*o.shape[:-2], -1),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )
        g = ttnn.linear(s, self.g_weight)
        g = ttnn.sigmoid(g)
        o = ttnn.multiply(o, g)
        x = ttnn.linear(o, self.o_weight)
        return x


class Transition(Module):
    def __init__(self, device: ttnn._ttnn.device.Device, state_dict: dict):
        super().__init__(device, state_dict)
        self.norm_weight = self.torch_to_tt("norm.weight", lambda x: x.expand(32, -1))
        self.norm_bias = self.torch_to_tt("norm.bias", lambda x: x.expand(32, -1))
        self.fc1_weight = self.torch_to_tt("fc1.weight")
        self.fc2_weight = self.torch_to_tt("fc2.weight")
        self.fc3_weight = self.torch_to_tt("fc3.weight")

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x_norm = ttnn.layer_norm(
            x, weight=self.norm_weight, bias=self.norm_bias, epsilon=1e-5
        )
        x_1 = ttnn.linear(x_norm, self.fc1_weight)
        x_1 = ttnn.silu(x_1)
        x_2 = ttnn.linear(x_norm, self.fc2_weight)
        x = ttnn.multiply(x_1, x_2)
        x = ttnn.linear(x, self.fc3_weight)
        return x


class PairformerLayer(Module):
    def __init__(
        self,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        att_head_dim: int,
        att_n_heads: int,
        device: ttnn._ttnn.device.Device,
        state_dict: dict,
    ):
        super().__init__(device, state_dict)
        self.triangle_multiplication_start = TriangleMultiplication(
            False, device, filter_dict(state_dict, "tri_mul_out")
        )
        self.triangle_multiplication_end = TriangleMultiplication(
            True, device, filter_dict(state_dict, "tri_mul_in")
        )
        self.triangle_attention_start = TriangleAttention(
            tri_att_head_dim,
            tri_att_n_heads,
            False,
            device,
            filter_dict(state_dict, "tri_att_start", "mha."),
        )
        self.triangle_attention_end = TriangleAttention(
            tri_att_head_dim,
            tri_att_n_heads,
            True,
            device,
            filter_dict(state_dict, "tri_att_end", "mha."),
        )
        self.attention_pair_bias = AttentionPairBias(
            att_head_dim,
            att_n_heads,
            True,
            device,
            filter_dict(state_dict, "attention"),
        )
        self.transition_z = Transition(device, filter_dict(state_dict, "transition_z"))
        self.transition_s = Transition(device, filter_dict(state_dict, "transition_s"))

    def __call__(
        self, s: ttnn.Tensor, z: ttnn.Tensor
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        z = ttnn.add(z, self.triangle_multiplication_start(z))
        z = ttnn.add(z, self.triangle_multiplication_end(z))
        z = ttnn.add(z, self.triangle_attention_start(z))
        z = ttnn.add(z, self.triangle_attention_end(z))
        z = ttnn.add(z, self.transition_z(z))
        s = ttnn.add(s, self.attention_pair_bias(s, z))
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
        device: ttnn._ttnn.device.Device,
        state_dict: dict,
    ):
        super().__init__(device, state_dict)
        self.blocks = [
            PairformerLayer(
                tri_att_head_dim,
                tri_att_n_heads,
                att_head_dim,
                att_n_heads,
                device,
                filter_dict(state_dict, f"layers.{i}"),
            )
            for i in range(n_blocks)
        ]

    def __call__(
        self, s: ttnn.Tensor, z: ttnn.Tensor
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        for block in self.blocks:
            s, z = block(s, z)
        return s, z


class PairformerModule(nn.Module):
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
        self.pairformer = None
        self.device = ttnn.open_device(device_id=0)
        ttnn.enable_program_cache(self.device)

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
        self.pairformer = Pairformer(
            self.n_blocks,
            self.tri_att_head_dim,
            self.tri_att_n_heads,
            self.att_head_dim,
            self.att_n_heads,
            self.device,
            filter_dict(state_dict, prefix[:-1]),
        )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor = None,
        pair_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return tuple(
            ttnn.to_torch(x).to(torch.float32)
            for x in self.pairformer(
                ttnn.from_torch(
                    s,
                    device=self.device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                ),
                ttnn.from_torch(
                    z,
                    device=self.device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                ),
            )
        )
