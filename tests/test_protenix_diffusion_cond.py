"""On-device parity for Protenix-v2 DiffusionConditioning (pair + single paths).
Pair path reproduces golden pair_z exactly (noise-independent); single path uses the
Fourier noise embedding. Gated on golden pkls (regenerate via scripts/protenix_extract_*)."""
import os, pickle, pytest, torch, torch.nn.functional as F, ttnn

_CKPT = "/home/ttuser/protenix_ckpt/protenix-v2.pt"
_TRUNK = os.path.expanduser("~/protenix_trunk_gold.pkl")
_TRUNKIN = os.path.expanduser("~/protenix_trunkin_gold.pkl")
_REF = os.path.expanduser("~/protenix_ref_out.pkl")
_DIFF = os.path.expanduser("~/protenix_diffusion_consistent.pkl")
pytestmark = pytest.mark.skipif(
    not all(os.path.exists(p) for p in (_CKPT, _TRUNK, _TRUNKIN, _REF, _DIFF)),
    reason="v2 ckpt or golden pkls missing")


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def _setup():
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from protenix_reference import remap_transition
    from tt_bio.tenstorrent import get_device, Transition, CORE_GRID_MAIN
    ck = torch.load(_CKPT, map_location="cpu", weights_only=True); ck = ck.get("model", ck)
    P = "module.diffusion_module.diffusion_conditioning."
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    return ck, P, dev, ckc, remap_transition, Transition, CORE_GRID_MAIN


def test_diffusion_conditioning_pair_on_device():
    ck, P, dev, ckc, remap_transition, Transition, CORE = _setup()
    g = lambda k: ck[P + k]; sub = lambda q: {k[len(P + q) + 1:]: v for k, v in ck.items() if k.startswith(P + q)}
    z_trunk = pickle.load(open(_TRUNK, "rb"))["z"].float()
    relp = pickle.load(open(_TRUNKIN, "rb"))["relp"]
    pair_z_gold = pickle.load(open(_REF, "rb"))["intermediates"]["diffusion_module"]["kwargs"]["pair_z"].float()
    T = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    lin = lambda x, w: ttnn.linear(x, T(w.t().contiguous()), compute_kernel_config=ckc, core_grid=CORE)
    relpe = lin(T(relp), g("relpe.linear_no_bias.weight"))
    zc = ttnn.layer_norm(ttnn.concat([T(z_trunk), relpe], dim=-1), weight=T(g("layernorm_z.weight")), epsilon=1e-5, compute_kernel_config=ckc)
    pz = ttnn.reshape(lin(zc, g("linear_no_bias_z.weight")), (1, 38, 38, 256))
    for nm in ("transition_z1", "transition_z2"):
        t = Transition(remap_transition(sub(nm)), ckc); pz = ttnn.add(pz, ttnn.reshape(t(pz), tuple(pz.shape)))
    out = torch.Tensor(ttnn.to_torch(pz)).float().reshape(pair_z_gold.shape)
    assert _pcc(out, pair_z_gold) > 0.99


def test_diffusion_conditioning_single_on_device():
    ck, P, dev, ckc, remap_transition, Transition, CORE = _setup()
    g = lambda k: ck[P + k]; sub = lambda q: {k[len(P + q) + 1:]: v for k, v in ck.items() if k.startswith(P + q)}
    dg = pickle.load(open(_DIFF, "rb"))["cond"]
    t_hat = dg["in"][0].float(); s_trunk = dg["kwargs"]["s_trunk"].float(); s_inputs = dg["kwargs"]["s_inputs"].float()
    s_single_gold = dg["out"][0].float()
    T = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    lin = lambda x, w: ttnn.linear(x, T(w.t().contiguous()), compute_kernel_config=ckc, core_grid=CORE)
    ss = lin(ttnn.layer_norm(T(torch.cat([s_trunk, s_inputs], -1)), weight=T(g("layernorm_s.weight")), epsilon=1e-5, compute_kernel_config=ckc), g("linear_no_bias_s.weight"))
    tp = torch.log(t_hat / 16.0) / 4
    fou = torch.cos(2 * torch.pi * (tp.unsqueeze(-1) * g("fourier_embedding.w") + g("fourier_embedding.b")))
    nn_ = lin(ttnn.layer_norm(T(fou), weight=T(g("layernorm_n.weight")), epsilon=1e-5, compute_kernel_config=ckc), g("linear_no_bias_n.weight"))
    ss = ttnn.reshape(ttnn.add(ss, nn_), (1, 38, 384))
    for nm in ("transition_s1", "transition_s2"):
        t = Transition(remap_transition(sub(nm)), ckc); ss = ttnn.add(ss, ttnn.reshape(t(ss), tuple(ss.shape)))
    out = torch.Tensor(ttnn.to_torch(ss)).float().reshape(s_single_gold.shape)
    assert _pcc(out, s_single_gold) > 0.99
