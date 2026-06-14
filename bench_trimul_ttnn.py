"""Benchmark the ttnn TriangleMultiplication module on Tenstorrent device 1.

Run:  TT_VISIBLE_DEVICES=1 python3 bench_trimul_ttnn.py [N] [D] [H] [iters]
(TT_VISIBLE_DEVICES=1 must be set BEFORE this imports ttnn -> device 1 = logical 0.)
"""
import os
import sys
import time

os.environ.setdefault("TT_VISIBLE_DEVICES", "1")

import torch
import ttnn
import tt_boltz.tenstorrent as T

N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
D = int(sys.argv[2]) if len(sys.argv) > 2 else 128   # d_pair
H = int(sys.argv[3]) if len(sys.argv) > 3 else 128   # hidden
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 20

device = T.get_device()
ckc = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

# Random weights matching TriangleMultiplication.__init__ key/shape expectations.
sd = {
    "norm_in.weight": torch.ones(D),
    "norm_in.bias": torch.zeros(D),
    "norm_out.weight": torch.ones(H),
    "norm_out.bias": torch.zeros(H),
    "g_in.weight": torch.randn(2 * H, D) * 0.1,
    "p_in.weight": torch.randn(2 * H, D) * 0.1,
    "g_out.weight": torch.randn(D, D) * 0.1,
    "p_out.weight": torch.randn(D, H) * 0.1,
}

trimul = T.TriangleMultiplication(ending=False, state_dict=sd, compute_kernel_config=ckc)

x_torch = torch.randn(1, N, N, D, dtype=torch.float32) * 0.5
x = ttnn.from_torch(x_torch, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

# Warm up (compiles kernels / fills program cache).
for _ in range(3):
    y = trimul(x)
    ttnn.deallocate(y)
ttnn.synchronize_device(device)

t0 = time.perf_counter()
for _ in range(ITERS):
    y = trimul(x)
    ttnn.deallocate(y)
ttnn.synchronize_device(device)
t1 = time.perf_counter()

ms = (t1 - t0) / ITERS * 1e3
print(f"[ttnn TriangleMultiplication] N={N} D={D} H={H}  {ms:.3f} ms/call  (warm, {ITERS} iters)")
