"""On-device parity for the full 48-block v2 Pairformer stack vs the REAL captured
trunk I/O (~/protenix_ref_out.pkl intermediates). Uses the raw v2 checkpoint +
remap_pairformer_block (pure dict remap) — no protenix install needed in sys python3.
"""
import os, re, pickle, pytest, torch, ttnn

_CKPT = "/home/ttuser/protenix_ckpt/protenix-v2.pt"
_GOLD = os.path.expanduser("~/protenix_ref_out.pkl")
pytestmark = pytest.mark.skipif(not (os.path.exists(_CKPT) and os.path.exists(_GOLD)),
                                reason="v2 ckpt or golden forward pkl missing")


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def test_v2_pairformer_stack_on_device():
    from protenix_reference import remap_pairformer_block
    from tt_bio.tenstorrent import get_device, Pairformer
    ck = torch.load(_CKPT, map_location="cpu", weights_only=True); ck = ck.get("model", ck)
    blk = lambda i: {k[len(f"module.pairformer_stack.blocks.{i}."):]: v
                     for k, v in ck.items() if k.startswith(f"module.pairformer_stack.blocks.{i}.")}
    nb = 1 + max(int(re.search(r"pairformer_stack\.blocks\.(\d+)\.", k).group(1))
                 for k in ck if "pairformer_stack.blocks." in k)
    b0 = blk(0)
    c_s = b0["single_transition.layernorm1.weight"].shape[0]
    nhp = b0["tri_att_start.linear.weight"].shape[0]
    chpa = b0["tri_att_start.mha.linear_q.weight"].shape[0] // nhp
    apb_nh = b0["attention_pair_bias.linear_nobias_z.weight"].shape[0]
    combined = {}
    for i in range(nb):
        for k, v in remap_pairformer_block(blk(i)).items():
            combined[f"layers.{i}.{k}"] = v
    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    pf = Pairformer(nb, chpa, nhp, c_s // apb_nh, apb_nh, True, combined, cfg)
    io = pickle.load(open(_GOLD, "rb"))["intermediates"]["pairformer_stack"]
    (s_in, z_in) = io["in"]; (s_out, z_out) = io["out"]
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    so, zo = pf(ft(s_in.unsqueeze(0)), ft(z_in.unsqueeze(0)))
    so = torch.Tensor(ttnn.to_torch(so)).float().reshape(s_out.shape)
    zo = torch.Tensor(ttnn.to_torch(zo)).float().reshape(z_out.shape)
    assert _pcc(so, s_out.float()) > 0.98 and _pcc(zo, z_out.float()) > 0.97
