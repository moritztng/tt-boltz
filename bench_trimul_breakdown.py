"""Phase breakdown of ttnn TriangleMultiplication on device 1, to compute the
full-op latency with the contraction replaced by the fused tt-metal kernel.

Faithfully replicates TriangleMultiplication.__call__ (tenstorrent.py) with
ttnn.synchronize_device + perf_counter timers around three phases:
  pre  = LN_in
  proj = gp_in projection (minimal_matmul) + sigmoid gate  (per chunk)
  cont = the contraction: _transform_chunk permutes + matmul + permute + concat
  post = LN_out + p_out/g_out projections + output gate

Run:  TT_VISIBLE_DEVICES=1 python3 bench_trimul_breakdown.py [N] [D] [H] [iters] [my_cont_ms]
"""
import os, sys, time
os.environ.setdefault("TT_VISIBLE_DEVICES", "1")
import torch, ttnn
import tt_boltz.tenstorrent as T

N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
D = int(sys.argv[2]) if len(sys.argv) > 2 else 128
H = int(sys.argv[3]) if len(sys.argv) > 3 else 128
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 30
MY_CONT_MS = float(sys.argv[5]) if len(sys.argv) > 5 else 0.116  # measured tt-metal per-channel kernel

dev = T.get_device()
ckc = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
sd = {
    "norm_in.weight": torch.ones(D), "norm_in.bias": torch.zeros(D),
    "norm_out.weight": torch.ones(H), "norm_out.bias": torch.zeros(H),
    "g_in.weight": torch.randn(2 * H, D) * 0.1, "p_in.weight": torch.randn(2 * H, D) * 0.1,
    "g_out.weight": torch.randn(D, D) * 0.1, "p_out.weight": torch.randn(D, H) * 0.1,
}
m = T.TriangleMultiplication(ending=False, state_dict=sd, compute_kernel_config=ckc)
x0 = ttnn.from_torch(torch.randn(1, N, N, D) * 0.5, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

def sync(): ttnn.synchronize_device(dev)

t = {"pre": 0.0, "proj": 0.0, "cont": 0.0, "post": 0.0}

def run_instrumented(x, acc):
    sync(); s = time.perf_counter()
    x_norm_in = ttnn.layer_norm(x, weight=m.in_norm_weight, bias=m.in_norm_bias,
                                epsilon=1e-5, compute_kernel_config=m.compute_kernel_config)
    Hh = x_norm_in.shape[1]
    mc = T._triangle_mul_memory_config(Hh)
    pc = T._triangle_mul_program_config((Hh + 31) // 32)
    if Hh > T.SEQ_LEN_MORE_CHUNKING:
        x_norm_in = ttnn.reallocate(x_norm_in)
    sync(); acc["pre"] += time.perf_counter() - s
    x = None
    for i in range(m.n_pairs):
        sync(); s = time.perf_counter()
        gp = ttnn.experimental.minimal_matmul(x_norm_in, m.gp_in_weight_fused_chunks[i],
                                              memory_config=mc, dtype=T._dtype(),
                                              compute_kernel_config=m.compute_kernel_config)
        g_a, g_b, p_a, p_b = ttnn.chunk(gp, chunks=4, dim=-1); ttnn.deallocate(gp)
        a = ttnn.multiply_(p_a, g_a, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        b = ttnn.multiply_(p_b, g_b, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(g_a); ttnn.deallocate(g_b)
        sync(); acc["proj"] += time.perf_counter() - s

        sync(); s = time.perf_counter()
        a = m._transform_chunk(a, (0, 3) + ((1, 2)), memory_config=mc)  # ending=False outgoing
        b = m._transform_chunk(b, (0, 3) + ((2, 1)), memory_config=mc)
        xc = ttnn.matmul(a, b, compute_kernel_config=m.compute_kernel_config,
                         memory_config=mc, program_config=pc, dtype=ttnn.bfloat16)
        ttnn.deallocate(a); ttnn.deallocate(b)
        xc = ttnn.permute(xc, (0, 2, 3, 1), memory_config=mc)
        if i == 0:
            x = ttnn.clone(xc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            xo = x; x = ttnn.concat([xo, xc], dim=-1); ttnn.deallocate(xo)
        ttnn.deallocate(xc)
        sync(); acc["cont"] += time.perf_counter() - s

    sync(); s = time.perf_counter()
    x = ttnn.layer_norm(x, weight=m.out_norm_weight, bias=m.out_norm_bias,
                        epsilon=1e-5, compute_kernel_config=m.compute_kernel_config)
    p_out = ttnn.linear(x, m.out_p_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=T._dtype(),
                        compute_kernel_config=m.compute_kernel_config, core_grid=T.CORE_GRID_MAIN)
    ttnn.deallocate(x)
    g_out = ttnn.linear(x_norm_in, m.g_out_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=T._dtype(),
                        compute_kernel_config=m.compute_kernel_config, core_grid=T.CORE_GRID_MAIN)
    ttnn.deallocate(x_norm_in)
    out = ttnn.multiply_(p_out, g_out, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    sync(); acc["post"] += time.perf_counter() - s
    ttnn.deallocate(out)

# warm
for _ in range(3):
    run_instrumented(x0, {"pre":0,"proj":0,"cont":0,"post":0})
# also time the un-instrumented full module for a clean reference
sync(); s = time.perf_counter()
for _ in range(ITERS):
    y = m(x0); ttnn.deallocate(y)
sync()
full_ms = (time.perf_counter() - s) / ITERS * 1e3

for _ in range(ITERS):
    run_instrumented(x0, t)
for k in t: t[k] = t[k] / ITERS * 1e3

phase_sum = sum(t.values())
ttnn_cont = t["cont"]
with_mine = phase_sum - ttnn_cont + MY_CONT_MS
print(f"\n=== N={N} D={D} H={H} ===")
print(f"ttnn full module (clean)      : {full_ms:.3f} ms")
print(f"ttnn phases  pre={t['pre']:.3f} proj={t['proj']:.3f} cont={t['cont']:.3f} post={t['post']:.3f}  sum={phase_sum:.3f} ms")
print(f"full op w/ MY contraction     : {with_mine:.3f} ms   (replaced cont {ttnn_cont:.3f} -> {MY_CONT_MS:.3f})")
print(f"FULL-OP SPEEDUP (clean/with-mine): {full_ms/with_mine:.2f}x   (phase-sum/with-mine: {phase_sum/with_mine:.2f}x)")
