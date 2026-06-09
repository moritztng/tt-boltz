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
