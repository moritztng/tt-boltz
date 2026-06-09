"""On-device parity for Protenix-v2 ConfidenceHead pae/pde heads (exact-validated, PCC 1.0)
+ 4-block confidence Pairformer. plddt/resolved are bf16-precision-sensitive (per-atom
einsum on 2-bin/50-bin low-dynamic-range outputs) and validated separately in the plan."""
import os, re, pickle, pytest, torch, torch.nn.functional as F, ttnn

_CKPT = "/home/ttuser/protenix_ckpt/protenix-v2.pt"
_CONF = os.path.expanduser("~/protenix_confidence_pre.pkl")
pytestmark = pytest.mark.skipif(not (os.path.exists(_CKPT) and os.path.exists(_CONF)),
                                reason="v2 ckpt or confidence golden pkl missing")


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def test_confidence_pae_pde_on_device():
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from protenix_reference import remap_pairformer_block
    from tt_bio.tenstorrent import get_device, Pairformer, CORE_GRID_MAIN as CORE
    ck = torch.load(_CKPT, map_location="cpu", weights_only=True); ck = ck.get("model", ck)
    P = "module.confidence_head."; g = lambda k: ck[P + k].float()
    kw = pickle.load(open(_CONF, "rb"))["kwargs"]
    feat = kw["input_feature_dict"]; s_inputs = kw["s_inputs"].float(); s_trunk = kw["s_trunk"].float()
    z_trunk = kw["z_trunk"].float(); x_pred = kw["x_pred_coords"].float()
    pae_g = pickle.load(open(_CONF, "rb"))["out"][1].float(); pde_g = pickle.load(open(_CONF, "rb"))["out"][2].float()
    N = s_inputs.reshape(-1, 449).shape[0]
    s_inputs = s_inputs.reshape(N, 449); s_trunk = s_trunk.reshape(N, 384); z_trunk = z_trunk.reshape(N, N, 256)
    hb = (P + "input_strunk_ln.bias") in ck
    s_t = F.layer_norm(torch.clamp(s_trunk, -512, 512), (384,)) * g("input_strunk_ln.weight") + (g("input_strunk_ln.bias") if hb else 0)
    mask = feat["distogram_rep_atom_mask"].bool()
    xr = x_pred.reshape(-1, 3)[mask]
    z = z_trunk + F.linear(s_inputs, g("linear_no_bias_s1.weight")).unsqueeze(1) + F.linear(s_inputs, g("linear_no_bias_s2.weight")).unsqueeze(0)
    d = torch.cdist(xr, xr); lb = g("lower_bins"); ub = g("upper_bins")
    oh = ((d.unsqueeze(-1) >= lb) & (d.unsqueeze(-1) < ub)).float()
    z = z + F.linear(oh, g("linear_no_bias_d.weight")) + F.linear(d.unsqueeze(-1), g("linear_no_bias_d_wo_onehot.weight"))
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    nb = 1 + max(int(re.search(r"pairformer_stack\.blocks\.(\d+)\.", k).group(1)) for k in ck if k.startswith(P + "pairformer_stack.blocks."))
    comb = {}
    for i in range(nb):
        bsd = {k[len(P + f"pairformer_stack.blocks.{i}."):]: v for k, v in ck.items() if k.startswith(P + f"pairformer_stack.blocks.{i}.")}
        for kk, vv in remap_pairformer_block(bsd).items():
            comb[f"layers.{i}.{kk}"] = vv
    b0 = P + "pairformer_stack.blocks.0."; nhp = ck[b0 + "tri_att_start.linear.weight"].shape[0]
    chpa = ck[b0 + "tri_att_start.mha.linear_q.weight"].shape[0] // nhp; apb_nh = ck[b0 + "attention_pair_bias.linear_nobias_z.weight"].shape[0]
    pf = Pairformer(nb, chpa, nhp, 384 // apb_nh, apb_nh, True, comb, ckc)
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    _, zo = pf(ft(s_t.unsqueeze(0)), ft(z.unsqueeze(0)))
    zf = torch.Tensor(ttnn.to_torch(zo)).float().reshape(N, N, 256)
    hpae = (P + "pae_ln.bias") in ck
    pae = F.linear(F.layer_norm(zf, (256,)) * g("pae_ln.weight") + (g("pae_ln.bias") if hpae else 0), g("linear_no_bias_pae.weight"))
    pde = F.linear(F.layer_norm(zf + zf.transpose(0, 1), (256,)) * g("pde_ln.weight") + (g("pde_ln.bias") if (P + "pde_ln.bias") in ck else 0), g("linear_no_bias_pde.weight"))
    assert _pcc(pae, pae_g) > 0.99 and _pcc(pde, pde_g) > 0.99
