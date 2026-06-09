"""On-device parity for the Protenix-v2 atom featurization (c_l, p_lm).

Validates tt_bio.protenix.AtomFeaturization against golden tensors extracted from
the real v2 reference (scripts/protenix_extract_atomfeat.py ->
~/protenix_atomfeat_gold.pkl). Decoupled from any protenix install in system
python3 — only needs the golden pkl + ttnn.
"""
import os, pickle, pytest, torch, ttnn

_GOLD = os.path.expanduser("~/protenix_atomfeat_gold.pkl")
pytestmark = pytest.mark.skipif(not os.path.exists(_GOLD), reason="golden pkl missing; run scripts/protenix_extract_atomfeat.py")


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def test_atom_featurization_on_device():
    from tt_bio.tenstorrent import get_device
    from tt_bio.protenix import AtomFeaturization

    g = pickle.load(open(_GOLD, "rb"))
    W, I = g["weights"], g["inputs"]
    dev = get_device()
    ck = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)

    def tt(x):
        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    feat = AtomFeaturization(W, ck)
    N = I["ref_pos"].shape[0]
    f_in = torch.cat([I["ref_mask"].reshape(N, 1), I["ref_element"].reshape(N, 128),
                      I["ref_atom_name_chars"].reshape(N, 256)], dim=-1)
    c_l = ttnn.to_torch(feat.c_l(tt(I["ref_pos"]), tt(torch.arcsinh(I["ref_charge"]).reshape(N, 1)),
                                 tt(I["ref_mask"].reshape(N, 1)), tt(f_in))).float()[:N]

    nb, nq, nk, _ = I["d_lm"].shape
    M = nb * nq * nk
    d = I["d_lm"].reshape(M, 3); v = I["v_lm"].reshape(M, 1)
    invd = (1.0 / (1.0 + (I["d_lm"] ** 2).sum(-1, keepdim=True))).reshape(M, 1)
    mt = g["mask_trunked"].reshape(M, 1)
    p_lm = ttnn.to_torch(feat.p_lm(tt(d), tt(v), tt(invd), tt(mt))).float()[:M].reshape(nb, nq, nk, 16)

    assert _pcc(c_l, g["golden_c_l"]) > 0.999
    assert _pcc(p_lm, g["golden_p_lm"]) > 0.999
