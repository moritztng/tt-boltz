"""On-device parity for the Protenix-v2 diffusion atom decoder vs real golden.
Gated on the pre-mutation golden pkl (scripts/protenix_extract_atomdec_pre.py).
Mirrors the trunk-component tests. The atom encoder (has_coords) is similarly
validated by scripts/protenix_atomenc_coords_parity.py (PCC 0.99999)."""
import os, pickle, pytest, torch, ttnn

_CKPT = "/home/ttuser/protenix_ckpt/protenix-v2.pt"
_DEC = os.path.expanduser("~/protenix_atomdec_pre.pkl")
_ENC = os.path.expanduser("~/protenix_atomenc_pre.pkl")
pytestmark = pytest.mark.skipif(
    not (os.path.exists(_CKPT) and os.path.exists(_DEC) and os.path.exists(_ENC)),
    reason="v2 ckpt or pre-mutation golden pkls missing (run scripts/protenix_extract_atomdec_pre.py)")


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def test_diffusion_atom_decoder_on_device():
    from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN as CORE
    from tt_bio.protenix import AtomTransformer
    ck = torch.load(_CKPT, map_location="cpu", weights_only=True); ck = ck.get("model", ck)
    P = "module.diffusion_module.atom_attention_decoder."; g = lambda k: ck[P + k]
    gd = pickle.load(open(_DEC, "rb"))["kwargs"]
    a2t = gd["atom_to_token_idx"].long()
    a = gd["a"].float(); a = a[0] if a.dim() == 3 else a
    q_skip = gd["q_skip"].float(); q_skip = q_skip[0] if q_skip.dim() == 3 else q_skip
    c_skip = gd["c_skip"].float(); c_skip = c_skip[0] if c_skip.dim() == 3 else c_skip
    p_skip = gd["p_skip"].float(); p_skip = p_skip[0] if p_skip.dim() == 5 else p_skip
    cg = pickle.load(open(_DEC, "rb"))["out"].float()
    N = q_skip.shape[0]; NT = a.shape[0]
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    T = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    lin = lambda x, w: ttnn.linear(x, T(w.t().contiguous()), compute_kernel_config=ckc, core_grid=CORE)
    S = torch.zeros(N, NT); S[torch.arange(N), a2t] = 1.0
    q = ttnn.add(ttnn.matmul(T(S), lin(T(a), g("linear_no_bias_a.weight")), compute_kernel_config=ckc, core_grid=CORE), T(q_skip))
    atx = AtomTransformer(3, {k[len(P + "atom_transformer."):]: v for k, v in ck.items() if k.startswith(P + "atom_transformer.")}, ckc)
    mt = pickle.load(open(_ENC, "rb"))["in"][8]["mask_trunked"].float()
    q_out = atx(ttnn.reshape(q, (1, N, 128)), ttnn.reshape(T(c_skip), (1, N, 128)), T(p_skip), mt)
    qn = ttnn.layer_norm(q_out, weight=T(g("layernorm_q.weight")), epsilon=1e-5, compute_kernel_config=ckc)
    coords = torch.Tensor(ttnn.to_torch(lin(qn, g("linear_no_bias_out.weight")))).float().reshape(1, N, 3)[:, :N]
    assert _pcc(coords, cg) > 0.99


def test_diffusion_atom_encoder_coords_on_device():
    from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN as CORE
    from tt_bio.protenix import AtomTransformer
    ck = torch.load(_CKPT, map_location="cpu", weights_only=True); ck = ck.get("model", ck)
    P = "module.diffusion_module.atom_attention_encoder."; g = lambda k: ck[P + k]
    dg = pickle.load(open(_ENC, "rb")); pin = dg["in"]; kw = dg["kwargs"]
    a2t = pin[0].long(); mt = pin[8]["mask_trunked"].float()
    r_l = kw["r_l"].float()[0]; s = kw["s"].float()[0]; c_l = kw["c_l"].float(); p_lm = kw["p_lm"].float()[0]
    a_gold = dg["out"][0].float()
    N = c_l.shape[0]; NT = int(a2t.max()) + 1; NQ, NK, PADL = 32, 128, 48
    NP = ((N + NQ - 1) // NQ) * NQ; nb = NP // NQ
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    T = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    lin = lambda x, w: ttnn.linear(x, T(w.t().contiguous()), compute_kernel_config=ckc, core_grid=CORE)
    s_proj = lin(ttnn.layer_norm(T(s), weight=T(g("layernorm_s.weight")), epsilon=1e-5, compute_kernel_config=ckc), g("linear_no_bias_s.weight"))
    S = torch.zeros(N, NT); S[torch.arange(N), a2t] = 1.0
    c_la = ttnn.add(T(c_l), ttnn.matmul(T(S), s_proj, compute_kernel_config=ckc, core_grid=CORE))
    q_l = ttnn.add(c_la, lin(T(r_l), g("linear_no_bias_r.weight")))

    def win_q(x):
        x = ttnn.to_layout(ttnn.reshape(x, (1, N, 128)), ttnn.ROW_MAJOR_LAYOUT); x = ttnn.pad(x, [[0, 0], [0, NP - N], [0, 0]], 0.0)
        return ttnn.to_layout(ttnn.reshape(x, (nb, NQ, 128)), ttnn.TILE_LAYOUT)

    def win_kv(x):
        x = ttnn.to_layout(ttnn.reshape(x, (1, N, 128)), ttnn.ROW_MAJOR_LAYOUT); Lp = PADL + NP + NK
        x = ttnn.pad(x, [[0, 0], [PADL, Lp - PADL - N], [0, 0]], 0.0)
        bl = [ttnn.slice(x, [0, i * NQ, 0], [1, i * NQ + NK, 128]) for i in range(nb)]
        return ttnn.to_layout(ttnn.reshape(ttnn.concat(bl, 0), (nb, NK, 128)), ttnn.TILE_LAYOUT)
    clq = ttnn.relu(win_q(c_la)); clk = ttnn.relu(win_kv(c_la))
    cl = ttnn.unsqueeze(lin(clq, g("linear_no_bias_cl.weight")), 2); cm = ttnn.unsqueeze(lin(clk, g("linear_no_bias_cm.weight")), 1)
    p = ttnn.add(ttnn.add(T(p_lm), cl), cm)
    m = lin(ttnn.relu(p), g("small_mlp.1.weight")); m = lin(ttnn.relu(m), g("small_mlp.3.weight")); m = lin(ttnn.relu(m), g("small_mlp.5.weight"))
    p = ttnn.add(p, m)
    atx = AtomTransformer(3, {k[len(P + "atom_transformer."):]: v for k, v in ck.items() if k.startswith(P + "atom_transformer.")}, ckc)
    q_out = atx(ttnn.reshape(q_l, (1, N, 128)), ttnn.reshape(c_la, (1, N, 128)), p, mt)
    q = ttnn.reshape(ttnn.relu(lin(q_out, g("linear_no_bias_q.weight"))), (N, 768))
    Mmat = torch.zeros(NT, N)
    for at in range(N):
        Mmat[a2t[at], at] = 1.0
    Mmat = Mmat / (Mmat.sum(-1, keepdim=True) + 1e-6)
    out = torch.Tensor(ttnn.to_torch(ttnn.matmul(T(Mmat), q, compute_kernel_config=ckc, core_grid=CORE))).float()[:NT].reshape(a_gold.shape)
    assert _pcc(out, a_gold.float()) > 0.99
