import ttnn, torch
from boltz.model.modules.tenstorrent import (
    filter_dict,
    device,
    TriangleAttention,
    TriangleAttentionGolden,
)

torch.set_grad_enabled(False)
torch.manual_seed(893)


def median_relative_error(a, b):
    return ((a - b).abs() / b.abs()).median().item()


state_dict = filter_dict(
    torch.load(
        "/home/moritz/.boltz/boltz2_conf.ckpt",
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )["state_dict"],
    "pairformer_module.layers.0.tri_att_start",
    "mha.",
)

triangle_attention = TriangleAttention(
    head_dim=32,
    n_heads=4,
    ending=False,
    state_dict=state_dict,
    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    ),
)

triangle_attention_golden = TriangleAttentionGolden(
    head_dim=32,
    n_heads=4,
    ending=False,
    state_dict=state_dict,
    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    ),
)

x = ttnn.from_torch(
    torch.randn(1, 512, 512, 128),
    device=device,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat16,
)

x_sdpa = triangle_attention(x)
x_golden = triangle_attention_golden(x)

mre = median_relative_error(ttnn.to_torch(x_sdpa), ttnn.to_torch(x_golden))
print("Median Relative Error:", mre)
assert mre < 0.1
print("Test passed")
