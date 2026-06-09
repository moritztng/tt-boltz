"""On-device parity for the full 4-block v2 MSA module vs the REAL captured trunk I/O.
Builds from the raw v2 ckpt via pure-dict remaps (protenix_reference) + golden I/O
(~/protenix_ref_out.pkl). Last block drops msa_stack (OPM + pair_stack only)."""
import os, pickle, pytest, torch, torch.nn.functional as F, ttnn

_CKPT = "/home/ttuser/protenix_ckpt/protenix-v2.pt"
_GOLD = os.path.expanduser("~/protenix_ref_out.pkl")
pytestmark = pytest.mark.skipif(not (os.path.exists(_CKPT) and os.path.exists(_GOLD)),
                                reason="v2 ckpt or golden forward pkl missing")


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def test_v2_msa_module_on_device():
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from protenix_reference import (remap_outer_product_mean, remap_pair_weighted_averaging,
                                     remap_transition, remap_msa_pair_stack)
    from tt_bio.tenstorrent import (get_device, OuterProductMean, PairWeightedAveraging,
                                    Transition, PairformerLayer)
    ck = torch.load(_CKPT, map_location="cpu", weights_only=True); ck = ck.get("model", ck)
    g = lambda k: ck["module.msa_module." + k]
    sub = lambda P: {k[len("module.msa_module." + P) + 1:]: v for k, v in ck.items()
                     if k.startswith("module.msa_module." + P)}
    io = pickle.load(open(_GOLD, "rb"))["intermediates"]["msa_module"]
    feat, z_in, s_inputs = io["in"][0], io["in"][1].float(), io["in"][2].float()
    z_gold = io["out"].float()
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    T = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    lin = lambda x, w: ttnn.linear(x, T(w.t().contiguous()), compute_kernel_config=ckc, core_grid=ttnn.CoreGrid(y=8, x=8))
    msa = F.one_hot(feat["msa"].long(), 32).float()
    ms = torch.cat([msa, feat["has_deletion"].unsqueeze(-1), feat["deletion_value"].unsqueeze(-1)], -1).unsqueeze(0)
    m = ttnn.add(lin(T(ms), g("linear_no_bias_m.weight")), lin(T(s_inputs), g("linear_no_bias_s.weight")))
    z = T(z_in.unsqueeze(0))
    has = lambda i: any(k.startswith(f"module.msa_module.blocks.{i}.msa_stack.") for k in ck)
    for i in range(4):
        P = f"blocks.{i}."
        opm = OuterProductMean(remap_outer_product_mean(sub(P + "outer_product_mean_msa")), ckc)
        pl = PairformerLayer(32, 8, None, None, False, remap_msa_pair_stack(sub(P + "pair_stack")), ckc)
        z = ttnn.add(z, opm(m, None, None))
        if has(i):
            pwa = PairWeightedAveraging(8, 8, remap_pair_weighted_averaging(sub(P + "msa_stack.msa_pair_weighted_averaging")), ckc)
            tm = Transition(remap_transition(sub(P + "msa_stack.transition_m")), ckc)
            m = ttnn.add(m, ttnn.reshape(pwa(m, ttnn.clone(z)), tuple(m.shape)))
            m = ttnn.add(m, ttnn.reshape(tm(m), tuple(m.shape)))
        z = pl(None, z)[1]
    zo = torch.Tensor(ttnn.to_torch(z)).float().reshape(z_gold.shape)
    assert _pcc(zo, z_gold) > 0.99
